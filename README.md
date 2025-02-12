<div align="center">

# Magic 1-For-1: Full-Attention DiT for Video Generation

</div>
<div align="center">
    <a href="https://magic-141.github.io/Magic-141/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
    <a href=""><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://arxiv.org/abs/2211.11018"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a> &ensp;
</div>

## 📖 Overview

**Magic 1-For-1** is an efficient image-to-video generation model designed to optimize memory usage and reduce inference latency. It decomposes the text-to-video generation task into two sub-tasks: **text-to-image generation** and **image-to-video generation**, enabling more efficient training and distillation.

### Updates
- **$\texttt{[2025-02]}$:** 🔥 We have released the technical report, model weights, and code for **Magic 1-For-1**.
- 🚀 More to Come!
We are continuously working on improving and expanding the capabilities of **Magic 1-For-1**. Contributions and collaborations are welcome! Join us in advancing the field of **interactive foundation video generation**.

## 📹 Demo


https://github.com/user-attachments/assets/94069a93-b2bb-4900-84f7-ca7603c04ecc


## 🛠️ Preparations

### Environment Setup
First, make sure **git-lfs** is installed (https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

It's recommended to use conda to manage the project's dependencies. First, it is needed to create a conda environment named video-generation and specify the Python version.
```bash
conda create -n video_infer python=3.9  # Or your preferred Python version
conda activate video_infer
```
The project's dependencies are listed in the requirements.txt file. You can use pip to install all dependencies at once.
```bash
pip install -r requirements.txt
```

### 📥 Downloading Model Weights

1. Create a Directory for Weights:
Create a directory to store the pretrained weights:

```bash
mkdir pretrained_weights
```

2. Download Magic 1-For-1 Weights:
Download the model weights by replacing `<model_weights_url>` with the actual URL:

```bash
wget -O pretrained_weights/magic_1_for_1_weights.pth <model_weights_url>
```

3. Download Hugging Face Components:
Use the Hugging Face CLI to download additional components: This will download the VAE, text encoder, the second text encoder, and the Llava VLM text encoder to the pretrained_weights directory.  

```bash
huggingface-cli download tencent/HunyuanVideo --local_dir pretrained_weights --local_dir_use_symlinks False
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local_dir pretrained_weights/text_encoder --local_dir_use_symlinks False
huggingface-cli download openai/clip-vit-large-patch14 --local_dir pretrained_weights/text_encoder_2 --local_dir_use_symlinks False
```

Make sure you have the `huggingface-cli` installed (`pip install huggingface_hub`). 

<!-- > ⚠️ **WARNING**: Flash Attention 3 is required for inference to avoid CUDA errors, by including `export USE_FLASH_ATTENTION3=1` -->

## 🚀 Inference 
### Text + Image to Video (Single GPU)
For Image + Text to Video generation, run the following command:

```bash
python test_ti2v.py --config configs/test/text_to_video/4_step_ti2v.yaml --quantization False
```

Alternatively, use the provided script:

```bash
bash scripts/run_flashatt3.sh
```

### 💻 Quantization
1. Install Optimum-Quanto:

```bash
pip install optimum-quanto
```

2. Enable Quantization:

Set `-quantization True` when running the script to enable quantization.

```bash
    python test_ti2v.py --config configs/test/text_to_video/4_step_ti2v.yaml --quantization True
```

### 🖥️ Multi-GPU Inference

To run inference on multiple GPUs, specify the number of GPUs and their IDs. Adjust the `ring_degree` and `ulysses_degree` values in the configuration file to match the number of GPUs used.

text, image to video
```bash
    bash scripts/run_flashatt3.sh test_ti2v.py configs/test/ti2v.yaml 1 0
```


## 📃 Citation

Please cite the following paper when using this model:
```bash
@article{yi2025magic,
  title={Magic 1-For-1: Generating One Minute Video Clips within One Minute},
  author={Hongwei Yi, Shitong Shao, Tian Ye, Jiantong Zhao, Qingyu Yin, Michael Lingelbach, Li Yuan, Yonghong Tian, Enze Xie, Daquan Zhou},
  journal={arXiv preprint arXiv:2211.11018},
  year={2025}
}
```

