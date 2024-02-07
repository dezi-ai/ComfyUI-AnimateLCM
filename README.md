# Comfyui-AnimateLCM

## Follow us: [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/dezi_labs.svg?style=social&label=Follow%20%40Dezi%20AI)](https://twitter.com/dezi_labs)



[Comfyui](https://github.com/comfyanonymous/ComfyUI) implementation for [AnimateLCM](https://animatelcm.github.io/) [[paper](https://arxiv.org/abs/2402.00769)].




<details>
<summary><b>Abstract</b></summary>
Video diffusion models has been gaining increasing attention for its ability to produce videos that are both coherent and of high fidelity. However, the iterative denoising process makes it computationally intensive and time-consuming, thus limiting its applications. Inspired by the Consistency Model (CM) that distills pretrained image diffusion models to accelerate the sampling with minimal steps and its successful extension Latent Consistency Model (LCM) on conditional image generation, we propose AnimateLCM, allowing for high-fidelity video generation within minimal steps. Instead of directly conducting consistency learning on the raw video dataset, we propose a decoupled consistency learning strategy that decouples the distillation of image generation priors and motion generation priors, which improves the training efficiency and enhance the generation visual quality. Additionally, to enable the combination of plug-and-play adapters in stable diffusion community to achieve various functions (e.g., ControlNet for controllable generation). we propose an efficient strategy to adapt existing adapters to our distilled text-conditioned video consistency model or train adapters from scratch without harming the sampling speed. We validate the proposed strategy in image-conditioned video generation and layout-conditioned video generation, all achieving top-performing results. Experimental results validate the effectiveness of our proposed method. Code and weights will be made public. More details are available at this https URL.
</details>

## Acknowledgement

This work is built on [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) but focus more on the accelearation of AnimateDiff text to video (t2v) generation.

- [https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
- [https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

Thanks the author: [Jedrzej Kosinski](https://github.com/Kosinkadink)

## Workflow

<b>Download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!</b>


| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/raw/main/workflows/videos/069c6cf5-103f-4f5d-ab3c-8d1d6977727e/069c6cf5-103f-4f5d-ab3c-8d1d6977727e-step10_00001.mp4"> | <video src="./workflows/videos/069c6cf5-103f-4f5d-ab3c-8d1d6977727e/069c6cf5-103f-4f5d-ab3c-8d1d6977727e-step10_00001.mp4"> |   <video src="./workflows/videos/069c6cf5-103f-4f5d-ab3c-8d1d6977727e/069c6cf5-103f-4f5d-ab3c-8d1d6977727e-step5_00001.mp4">    |
