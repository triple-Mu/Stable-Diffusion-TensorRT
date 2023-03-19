from pathlib import Path
from typing import Tuple, Union

import onnx
import torch
from torch import Module, Tensor

from trt_sd.utils import VAEDecoderWrapper, VAEEncoderWrapper

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    diffusers = None
    print('Using "pip install diffusers" to install diffusers')
    exit()

from io import BytesIO

from diffusers.models.attention import GEGLU, CrossAttention

MAJOR, MINOR = map(int, torch.__version__.split('.')[:2])
OPSET = 16


class TorchStableDiffusionModel(Module):

    def __init__(self,
                 model_name_or_path: str = 'CompVis/stable-diffusion-v1-4',
                 dtype: torch.dtype = torch.float32,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__()
        pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path,
                                                       revision='main',
                                                       torch_dtype=dtype,
                                                       use_auth_token=True)
        self.pipe = pipe.to(device)
        self.device = device
        self.is_switch = False

    def get_clip(self, fp16=False) -> Module:
        _clip = self.pipe.text_encoder
        _clip.eval()
        if fp16:
            _clip = _clip.half()
        return _clip

    def get_unet(self, fp16=True) -> Module:
        _unet = self.pipe.unet
        _unet.eval()
        if fp16:
            _unet = _unet.half()
        return _unet

    def get_vae_encoder(self, fp16=False) -> Module:
        _vae_encoder = VAEEncoderWrapper(self.pipe.vae)
        _vae_encoder.eval()
        if fp16:
            _vae_encoder = _vae_encoder.half()
        return _vae_encoder

    def get_vae_decoder(self, fp16=False) -> Module:
        _vae_decoder = VAEDecoderWrapper(self.pipe.vae)
        _vae_decoder.eval()
        if fp16:
            _vae_decoder = _vae_decoder.half()
        return _vae_decoder

    def get_clip_input(self, text: str) -> Tensor:
        clip_input = self.pipe.tokenizer(
            text,
            padding='max_length',
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids.to(self.device)
        return clip_input

    def get_unet_inputs(self, latents: Tensor, timestep: Tensor,
                        encoder_hidden_state: Tensor, fp16: bool = False) -> \
            Tuple[Tensor, Tensor, Tensor]:
        latents = latents.to(self.device)
        timestep = timestep.to(self.device)
        encoder_hidden_state = encoder_hidden_state.to(self.device)
        if fp16:
            latents = latents.half()
            timestep = timestep.half()
            encoder_hidden_state = encoder_hidden_state.half()
        return latents, timestep, encoder_hidden_state

    def get_random_unet_inputs(self, text: str,
                               height: int = 64, width: int = 64,
                               batch: int = 2, fp16: bool = True) -> \
            Tuple[Tensor, Tensor, Tensor]:
        _clip = self.get_clip()
        _clip_input = self.get_clip_input(text)
        unet_input = _clip(_clip_input)[0]
        uncond_input = self.pipe.tokenizer(
            [''],
            padding='max_length',
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors='pt').input_ids.to(self.device)
        uncond_embeddings = _clip(uncond_input)[0]

        latents = torch.randn(batch, 4, height, width).to(self.device)
        timestep = torch.tensor(1, dtype=torch.float32).to(self.device)
        encoder_hidden_state = torch.cat([uncond_embeddings, unet_input])
        if fp16:
            latents = latents.half()
            timestep = timestep.half()
            encoder_hidden_state = encoder_hidden_state.half()
        return latents, timestep, encoder_hidden_state

    def get_vae_deccoder_input(self,
                               latents: Tensor,
                               fp16: bool = True) -> Tensor:
        latents = latents.to(self.device)
        if fp16:
            latents = latents.half()
        return latents

    def get_random_vae_deccoder_input(self,
                                      height: int = 64,
                                      width: int = 64,
                                      batch: int = 2,
                                      fp16: bool = False) -> Tensor:
        latents = torch.randn(batch, 4, height, width).to(self.device)
        if fp16:
            latents = latents.half()
        return latents

    def switch_to_deploy(self) -> None:
        if self.is_switch:
            return
        if MINOR < 13:
            try:
                from torch.onnx.symbolic_registry import register_op
            except ImportError:
                print('PyTorch version is wrong!')
                exit()
        else:
            try:
                from torch.onnx import \
                    register_custom_op_symbolic as register_op
            except ImportError:
                print('PyTorch version is wrong!')
                exit()

        from trt_sd.utils import (MyCrossAttnProcessor, cross_attn_forward_v1,
                                  cross_attn_forward_v2, group_norm,
                                  layer_norm, split_gelu)
        register_op('layer_norm', layer_norm, '', OPSET)
        register_op('group_norm', group_norm, '', OPSET)

        cpv1 = MyCrossAttnProcessor(cross_attn_forward_v1)
        cpv2 = MyCrossAttnProcessor(cross_attn_forward_v2)
        _unet = self.get_unet()
        for k, m in _unet.named_modules():
            tp = type(m)
            if tp is GEGLU:
                split_gelu(m)
            elif tp is CrossAttention:
                k = k.split('.')[-1]
                if k == 'attn1':
                    m.set_processor(cpv1)
                elif k == 'attn2':
                    m.set_processor(cpv2)

    @torch.no_grad()
    def export_clip(self, save_to: str, fp16=False):
        self.switch_to_deploy()
        _clip = self.get_clip(fp16=fp16)
        _clip_input = self.get_clip_input(
            'a photo of an astronaut riding a horse on mars')
        # forward once
        _clip(_clip_input)[0]
        torch.onnx.export(_clip,
                          _clip_input,
                          save_to,
                          input_names=['input_ids'],
                          output_names=['last_hidden_state', 'pooler_out'],
                          opset_version=OPSET)

    @torch.no_grad()
    def export_unet(self, save_to: str, fp16=True):
        self.switch_to_deploy()
        _unet = self.get_unet(fp16=fp16)
        latents, timestep, encoder_hidden_state = self.get_random_unet_inputs(
            'a photo of an astronaut riding a horse on mars', fp16=fp16)
        # forward once
        _unet(latents, timestep, encoder_hidden_state)
        with BytesIO() as f:
            if not fp16:
                f = Path('tmp_dir')
                f.mkdir(parents=True, exist_ok=True)
                f = str(f / 'unet.onnx')
            torch.onnx.export(
                _unet, (latents, timestep, encoder_hidden_state),
                f,
                input_names=['sample', 'timestep', 'encoder_hidden_state'],
                output_names=['out_sample'],
                opset_version=OPSET)
            if not isinstance(f, str):
                f.seek(0)
        onnx_model = onnx.load(f)
        if fp16:
            onnx.save(onnx_model, save_to)
        else:
            onnx.save(onnx_model,
                      save_to,
                      save_as_external_data=True,
                      all_tensors_to_one_file=True,
                      location=save_to + '.data',
                      convert_attribute=True)

    def export_vae_decoder(self, save_to: str, fp16=False):
        self.switch_to_deploy()
        _vae_decoder = self.get_vae_decoder(fp16=fp16)
        latents = self.get_random_vae_deccoder_input(fp16=fp16)
        _vae_decoder(latents)
        torch.onnx.export(_vae_decoder,
                          latents,
                          save_to,
                          input_names=['latents'],
                          output_names=['sample'],
                          opset_version=OPSET)
