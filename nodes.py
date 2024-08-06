import os
import torch
import gc
import folder_paths
import comfy.model_management as mm
import comfy.utils
from diffusers import (DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from .easyanimate.models.transformer3d import Transformer3DModel
from .easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from .easyanimate.aspect_ratios import ASPECT_RATIO_512, get_closest_ratio

script_directory = os.path.dirname(os.path.abspath(__file__))

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

class DownloadAndLoadEasyAnimateModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   
                        'EasyAnimateV3-XL-2-InP-512x512-bf16',
                        'EasyAnimateV3-XL-2-InP-768x768-bf16',
                        'EasyAnimateV3-XL-2-InP-960x960-bf16',
                    ],
                    ),
            "precision": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                    ], {
                        "default": 'bf16'
                    }),
            "low_gpu_memory_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("EASYANIMATEPIPE",)
    RETURN_NAMES = ("easyanimate_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "EasyAnimateWrapper"

    def loadmodel(self, precision, model, low_gpu_memory_mode):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = comfy.utils.ProgressBar(5)
        config_path  = os.path.join(script_directory, "config", "easyanimate_video_slicevae_motion_module_v3.yaml")
        config = OmegaConf.load(config_path)

        base_path = os.path.join(folder_paths.models_dir, "easyanimate")
        download_path = os.path.join(base_path, "transformer")
        transformers_path = os.path.join(base_path, "transformer", f"{model}.safetensors")
        
        if not os.path.exists(transformers_path):
            print(f"Downloading model to: {transformers_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/EasyAnimate-bf16", 
                                allow_patterns=[f"*{model}*"],
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)
            pbar.update(1)
            
        common_path = os.path.join(base_path, "common")

        if not os.path.exists(common_path):
            print(f"Downloading model to: {common_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"alibaba-pai/EasyAnimateV3-XL-2-InP-768x768", 
                                ignore_patterns=["transformer/*","text_encoder/*"],
                                local_dir=common_path, 
                                local_dir_use_symlinks=False)

            pbar.update(1)

        transformer_config = Transformer3DModel.load_config(os.path.join(script_directory, "config", "transformer_config.json"))
        transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        transformer_config.update(transformer_additional_kwargs)

        with (init_empty_weights()):
            self.transformer = Transformer3DModel.from_config(transformer_config)

        sd = comfy.utils.load_torch_file(os.path.join(transformers_path))
        for key in sd:
            set_module_tensor_to_device(self.transformer, key, dtype=dtype, device=device, value=sd[key])
        del sd

        pbar.update(1)

        vae = AutoencoderKLMagvit.from_pretrained(common_path, subfolder="vae").to(dtype)
        vae.upcast_vae = True

        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(common_path, subfolder="image_encoder").to(device, dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(common_path, subfolder="image_encoder")

        pbar.update(1)

        scheduler = DPMSolverMultistepScheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))
        pipeline = EasyAnimateInpaintPipeline(
            vae=vae,
            transformer=self.transformer,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
        )

        if low_gpu_memory_mode:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()
        
        pipeline_dict = {
            'pipeline': pipeline,
            'dtype': dtype,
            'model_name': model,
            'model_path': common_path,
        }
        pbar.update(1)
        return (pipeline_dict,)
    
class EasyAnimateTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, prompt):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.truncation = True
        clip.tokenizer.t5xxl.max_length = 120
        clip.cond_stage_model.t5xxl.return_attention_masks = True
        clip.cond_stage_model.t5xxl.use_attention_masks = True
        tokens = clip.tokenize(prompt.lower().strip(), return_word_ids=True)["t5xxl"]
        clip.cond_stage_model.t5xxl.to(device)
        embeds, _, attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
        clip.cond_stage_model.t5xxl.to(offload_device)
        embeds = {
            "embeds": embeds,
            "attention_mask": attention_mask['attention_mask']
        }

        return (embeds, )
        
class EasyAnimateSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "easyanimate_pipeline": ("EASYANIMATEPIPE",),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "width": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 8}),
            "height": ("INT", {"default": 672, "min": 64, "max": 2048, "step": 8}),
            "frames": ("INT", {"default": 16, "min": 8, "max": 144, "step": 8}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": ([
                "DPMSolverMultistepScheduler",
                "EulerDiscreteScheduler",
                "EulerAncestralDiscreteScheduler",
                "PNDMScheduler",
                "DDIMScheduler",
            ],),
           
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_start": ("IMAGE",),
                "image_end": ("IMAGE",),
               
            }
        }

    RETURN_TYPES = ("EASYANIMLATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, easyanimate_pipeline, positive, negative, width, height, frames, cfg, steps, seed, scheduler, keep_model_loaded, 
                image_start=None, image_end=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = easyanimate_pipeline['dtype']
        pipeline = easyanimate_pipeline['pipeline']
        

        model_name = easyanimate_pipeline['model_name']
        if "512" in model_name:
            base_resolution = 512
        elif "768" in model_name:
            base_resolution = 768
        elif "960" in model_name:
            base_resolution = 960

        vae = pipeline.vae
        video_length = int(frames // vae.mini_batch_encoder * vae.mini_batch_encoder) if frames != 1 else 1

        if image_start is not None:
            B, H, W, C = image_start.shape
            image_start = clip_image = image_start.permute(0, 3, 1, 2).to(device)
            aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            closest_size, closest_ratio = get_closest_ratio(H, W, ratios=aspect_ratio_sample_size)
            height, width = [int(x / 16) * 16 for x in closest_size]
            print("closest size: ", closest_size, " closest ratio: ", closest_ratio, " height: ", height, " width: ", width)

            pixels = comfy.utils.common_upscale(image_start, width, height, "lanczos", "disabled")

            input_video = torch.tile(
            pixels.unsqueeze(2),  # Permute B, H, W, C to B, C, H, W and unsqueeze T
            [1, 1, video_length, 1, 1]
            )
            print("input_video: ", input_video.shape)
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 1

            if image_end is not None:
                input_video[:, :, -1:] = image_end.permute(0, 3, 1, 2).unsqueeze(2).to(device)
                input_video_mask[:, :, -1:] = 0
        else:
            input_video = torch.zeros([1, 3, video_length, height, width])
            input_video_mask = torch.ones([1, 1, video_length, height, width])
            clip_image = None


        generator = torch.Generator(device=device).manual_seed(seed)
        model_path = easyanimate_pipeline['model_path']
        scheduler_classes = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
            "DDIMScheduler": DDIMScheduler,
        }

        if scheduler in scheduler_classes:
            noise_scheduler = scheduler_classes[scheduler].from_pretrained(model_path, subfolder='scheduler')
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        pipeline.scheduler = noise_scheduler

        sample = pipeline(
            video_length = video_length,
            prompt_embeds = positive["embeds"].to(dtype).to(device),
            prompt_attention_mask = positive["attention_mask"].to(dtype).to(device),
            negative_prompt_embeds = negative["embeds"].to(dtype).to(device),
            negative_prompt_attention_mask = negative["attention_mask"].to(dtype).to(device),
            height = height,
            width = width,
            generator = generator,
            guidance_scale = cfg,
            num_inference_steps = steps,
            video = input_video,
            mask_video = input_video_mask,
            clip_image = clip_image,
        )

        if not keep_model_loaded:
            pipeline.unet.to(offload_device)
            pipeline.vae.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        return sample,

class EasyAnimateTextEncodeOrig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_pipeline": ("EASYANIMATEPIPE", ),
                "prompt": ("STRING", {"multiline": True, "default": "",}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES =("conditioning", "negative_conditioning")
    FUNCTION = "encode"
    CATEGORY = "EasyAnimateWrapper"
    
    def encode(self, easyanimate_pipeline, prompt, negative_prompt):
        device = mm.get_torch_device()
        
        mm.unload_all_models()
        mm.soft_empty_cache()

        self.text_encoder = easyanimate_pipeline['pipeline'].text_encoder
        self.tokenizer = easyanimate_pipeline['pipeline'].tokenizer

        dtype = easyanimate_pipeline['dtype']

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = 120
        clean_caption = False
        prompt = EasyAnimateInpaintPipeline._text_preprocessing(self, prompt, clean_caption=clean_caption)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
       
        uncond_tokens = [negative_prompt] * batch_size
        uncond_tokens = EasyAnimateInpaintPipeline._text_preprocessing(self, uncond_tokens, clean_caption=clean_caption)
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_prompt_attention_mask = uncond_input.attention_mask
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
        
        #self.text_encoder.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()

        pos_embeds = {
            'embeds': prompt_embeds,
            'attention_mask': prompt_attention_mask,
        }
        neg_embeds = {
            'embeds': negative_prompt_embeds,
            'attention_mask': negative_prompt_attention_mask
        }
        
        return (pos_embeds, neg_embeds)
    
class EasyAnimateDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_pipeline": ("EASYANIMATEPIPE", ),
                "latents": ("EASYANIMLATENTS", ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "decode"
    CATEGORY = "EasyAnimateWrapper"
    
    def decode(self, easyanimate_pipeline, latents):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        mm.unload_all_models()
        mm.soft_empty_cache()
        self.vae = easyanimate_pipeline['pipeline'].vae

        latents = 1 / self.vae.config.scaling_factor * latents
        if self.vae.quant_conv.weight.ndim==5:
            mini_batch_encoder = self.vae.mini_batch_encoder
            mini_batch_decoder = self.vae.mini_batch_decoder
            if self.vae.slice_compression_vae:
                video = self.vae.decode(latents)[0]
            else:
                video = []
                pbar = comfy.utils.ProgressBar(latents.shape[2])
                for i in range(0, latents.shape[2], mini_batch_decoder):
                    with torch.no_grad():
                        start_index = i
                        end_index = i + mini_batch_decoder
                        latents_bs = self.vae.decode(latents[:, :, start_index:end_index, :, :])[0]
                        video.append(latents_bs)
                        pbar.update(1)
                video = torch.cat(video, 2)
            video = video.clamp(-1, 1)
            video = EasyAnimateInpaintPipeline.smooth_output(self, video, mini_batch_encoder, mini_batch_decoder).cpu().clamp(-1, 1)

        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().float()
        return (video,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadEasyAnimateModel": DownloadAndLoadEasyAnimateModel,
    "EasyAnimateSampler": EasyAnimateSampler,
    "EasyAnimateTextEncode": EasyAnimateTextEncode,
    "EasyAnimateDecode": EasyAnimateDecode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadEasyAnimateModel": "(Down)Load EasyAnimate Model",
    "EasyAnimateSampler": "EasyAnimate Sampler",
    "EasyAnimateTextEncode": "EasyAnimate Text Encode",
    "EasyAnimateDecode": "EasyAnimate Decode"
}
