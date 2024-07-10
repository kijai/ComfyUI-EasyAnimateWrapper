# WORK IN PROGRESS


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
│   ├───text_encoder
│   │       config.json
│   │       model.fp16-00001-of-00002.safetensors
│   │       model.fp16-00002-of-00002.safetensors
│   │       model.safetensors.index.json
│   │
│   ├───tokenizer
│   │       added_tokens.json
│   │       special_tokens_map.json
│   │       spiece.model
│   │       tokenizer_config.json
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
