# WORK IN PROGRESS
## Update:
Switched the text encoder to use the Comfy native T5 loader as the model is same that's used with SD3 and Flux.
It can be downloaded from the Manager, or from here, to `ComfyUI/models/clip`:

fp16: https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/blob/main/pytorch_model.safetensors

fp8: https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/blob/main/t5xxl_fp8_e4m3fn.safetensors



https://github.com/user-attachments/assets/a9e479dd-83fa-4558-abce-0d533210c66e

Everything should be autodownloaded, but for manual install:

bf16 pruned versions of the transformers models available here:

https://huggingface.co/Kijai/EasyAnimate-bf16/tree/main

Everything else from any of the V3 EasyAnimate repos:

https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768

For now the models are using the following structure to allow using the different transformers -models without downloading everything else multiple times.
```
ComfyUI/models/easyanimate/
├───common
│   │   .gitattributes
│   │   model_index.json
│   │   README.md
│   │
│   ├───image_encoder
│   │       config.json
│   │       preprocessor_config.json
│   │       pytorch_model.bin
│   │
│   ├───scheduler
│   │       scheduler_config.json
│   │
│   └───vae
│           config.json
│           diffusion_pytorch_model.safetensors
│
└───transformer
        EasyAnimateV3-XL-2-InP-512x512-bf16.safetensors
        EasyAnimateV3-XL-2-InP-768x768-bf16.safetensors
        EasyAnimateV3-XL-2-InP-960x960-bf16.safetensors
```
