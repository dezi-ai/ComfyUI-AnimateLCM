# Comfyui-AnimateLCM

## Follow us: [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/dezi_labs.svg?style=social&label=Follow%20%40Dezi%20AI)](https://twitter.com/dezi_labs)



[Comfyui](https://github.com/comfyanonymous/ComfyUI) implementation for [AnimateLCM](https://animatelcm.github.io/) [[paper](https://arxiv.org/abs/2402.00769)].




<details>
<summary><b>Abstract</b></summary>
Video diffusion models has been gaining increasing attention for its ability to produce videos that are both coherent and of high fidelity. However, the iterative denoising process makes it computationally intensive and time-consuming, thus limiting its applications. Inspired by the Consistency Model (CM) that distills pretrained image diffusion models to accelerate the sampling with minimal steps and its successful extension Latent Consistency Model (LCM) on conditional image generation, we propose AnimateLCM, allowing for high-fidelity video generation within minimal steps. Instead of directly conducting consistency learning on the raw video dataset, we propose a decoupled consistency learning strategy that decouples the distillation of image generation priors and motion generation priors, which improves the training efficiency and enhance the generation visual quality. Additionally, to enable the combination of plug-and-play adapters in stable diffusion community to achieve various functions (e.g., ControlNet for controllable generation). we propose an efficient strategy to adapt existing adapters to our distilled text-conditioned video consistency model or train adapters from scratch without harming the sampling speed. We validate the proposed strategy in image-conditioned video generation and layout-conditioned video generation, all achieving top-performing results. Experimental results validate the effectiveness of our proposed method. Code and weights will be made public. More details are available at this https URL.
</details>

## Installation

1. Install [Comfyui](https://github.com/comfyanonymous/ComfyUI)
2. Download AnimateLCM from huggingface [https://huggingface.co/wangfuyun/AnimateLCM/tree/main](https://huggingface.co/wangfuyun/AnimateLCM/tree/main)
3. Place `sd15_t2v_beta.ckpt` to ComfyUI `ComfyUI/models/animatediff_models` and place `sd15_lora_beta.safetensors` to `ComfyUI/models/loras`

## Statistics

> For 5 step, average generation time for the advanced flow is 21s, 10 step : 42s, 20 step: 86s

## Workflow

<b>Download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!</b>

### The pure workflow using animate-diff
> Run faster but with a quality trade-off

[AnimateLCM.json](./workflows/animatelcm.json)
<center>
<image src="./workflows/animatelcm.svg" width="50%">
</center>


### The advanced workflow using custom-sampler
[AnimateLCM_advanced.json](./workflows/animatelcm_advanced.json) [Reddit](https://www.reddit.com/r/comfyui/comments/1ajjp9v/animatelcm_support_just_dropped/)
<center>
<image src="./workflows/animatelcm_advanced.svg" width="50%">
</center>


---

> Prompt
<b>
mustle manly man holding a gun, elegant, dynamic pose, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, art by Artgerm and Greg Rutkowski and Alphonse Mucha
</b>

| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/0c509955-a702-4c76-97c5-bd382cdfed55"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/f69344fc-95f4-4284-966e-5ec94ac51fe3"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/44fdfe1f-2af2-4130-815a-88c326f35bee">    |


---

> Prompt
<b>
cute painting of a frog dressed as a detective. The frog has a magnifying glass in one hand and a hat similar to Sherlock Holmes highly stylized, matte coloring, childish look, on a page of an illustrated book for children, drawn with Photoshop
</b>


| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/965ba56e-bb5a-4130-ae54-e1c84601dced"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/37a46067-f0a9-46b7-ad1e-ca77cb72956f"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/59fdcd2f-3fbe-44cd-ae7c-7fa86aa65f47">    |

---

> Prompt
<b>
mechwarrior 5 : mercenaries mech megaman transformer robot boss tank engine game octane render, 4 k, hd 2 0 2 2 3 d cgi rtx hdr style chrome reflexion glow fanart, global illumination ray tracing hdr fanart arstation by ian pesty by jesper ejsing pixar and disney unreal zbrush central hardmesh
</b>




| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/8f525ec3-0152-49e5-ac5e-c57d60b38db0"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/250c9165-c625-4666-a5e4-ec84c7b72ac9"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/c9eb3afc-d032-4a68-838a-8352568dcf09">    |

---

> Prompt
<b>
a male anthromorphic cyborg dragon, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by john collier and albert aublet and krenz cushart and artem demura
</b>

| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/984a321c-cf62-482a-a0f5-40d692d29cb5"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/dc1219eb-2d5b-45c8-8e8e-7cde96e663d8"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/71829b4c-c143-4a14-b841-635d111a4be7">    |

---

> Prompt
<b>
full figure bella thorne, hyperrealistic portrait, bladerunner street, art of elysium and jeremy mann and alphonse mucha, fantasy art, photo realistic, dynamic lighting, artstation, poster, volumetric lighting, very detailed face, 4 k, award winning
</b>

| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/fce2a64e-e302-44a4-a37c-da4c10f5574b"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/d2537704-e9c0-4049-8f08-46af4d9550d9"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/a0d0d027-eb62-49fa-b79a-6733e21e76fa">    |


---

> Prompt
<b>
photographic portrait of a stunningly beautiful gothic female in soft dreamy light at sunset, contemporary fashion shoot, by edward robert hughes, annie leibovitz and steve mccurry, david lazar, jimmy nelsson, breathtaking, 8 k resolution, extremely detailed, beautiful, establishing shot, artistic, hyperrealistic, beautiful face, octane render
</b>

| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/8aa06a3e-e891-44a2-9ffa-c30fe9e005c1"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/1beb172d-7646-4a9f-9067-a5ceea8075c3"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/2f552efc-d95c-4005-95ed-c212d8c502e1">    |
















---

> Prompt
<b>
realistic ethereal hydrangea dryad wearing beautiful dress, deity of hydrangeas made of hydrangeas, mystical, 4k digital masterpiece by Alberto Seveso and Anna Dittman, Ruan Jia, rossdraws, full view, fantasycore, Hyperdetailed, realistic oil on linen, soft lighting, Iconography background, featured on Artstation
</b>

| LCM step=5                                                   | LCM step = 10                                                |  LCM step = 20    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/6bf0fefa-7deb-4811-8339-13e156320cc4"> | <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/b1649f7a-36fb-44c2-827c-68661faf52a4"> |   <video src="https://github.com/dezi-ai/ComfyUI-AnimateLCM/assets/154349745/8e0c0cf3-acdd-4102-83cd-588c2f6a4202">    |























## Acknowledgement

This work is built on [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved), [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) and [ComfyUI-sampler-lcm-alternative](https://github.com/jojkaart/ComfyUI-sampler-lcm-alternative) but focus more on the accelearation of AnimateDiff text to video (t2v) generation.

- [https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
- [https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [https://github.com/jojkaart/ComfyUI-sampler-lcm-alternative](https://github.com/jojkaart/ComfyUI-sampler-lcm-alternative)



























