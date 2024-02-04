import os
import copy
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from animatelcm.models.unet import UNet3DConditionModel
from animatelcm.pipline import AnimationPipeline
import folder_paths
import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from safetensors import safe_open
from animatelcm.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.utils.lcm_utils import convert_lcm_lora
from diffusers.utils.import_utils import is_xformers_available


scheduler_dict = {
    "LCM": LCMScheduler,
}

class AnimateLCMModelLoader:
    def __init__(self):
        self.models = {}

    @classmethod
    def INPUT_TYPES(s):
        animatelcm_checkpoints = folder_paths.get_filename_list("animatelcm")
        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")

        return {
            "required": {
                "motion_module" : (animatelcm_checkpoints ,{
                    "default" : animatelcm_checkpoints[0]
                }),
                "DreamBooth" : (animatelcm_checkpoints ,{
                    "default" : animatelcm_checkpoints[0]
                }),
                "LCM_LORA": (animatelcm_checkpoints ,{
                    "default" : animatelcm_checkpoints[0]
                }),
                "LCM_LORA_weight": ("FLOAT", {
                    "default": 0.8,
                    "step": 0.05,
                    "min": 0,
                    "max": 1.0,
                    "display": "slider",
                }),
                "sampler": (["LCM"],),
                "device" : (devices,),
            },

        }

    RETURN_TYPES = ("AnimateLCM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "dezi-ai"

    def load_model(self, motion_module, DreamBooth, LCM_LORA, LCM_LORA_weight, sampler, device):
        all_model_base = folder_paths.get_folder_paths("animatelcm")[0]
        stable_diffusion_paths = os.path.join(all_model_base, "StableDiffusion", "stable-diffusion-v1-5")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_paths, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_paths, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(
            stable_diffusion_paths, subfolder="vae").cuda()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.inference_config  = OmegaConf.load(os.path.join(current_dir, "configs", "inference.yaml"))
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            stable_diffusion_paths, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()

        # load motion model
        if self.unet is None:
            print("Please select a pretrained model path.")
            raise ValueError("No diffusion model is selected.")
        else:
            motion_module_dir = os.path.join(
                all_model_base, motion_module)
            motion_module_state_dict = torch.load(
                motion_module_dir, map_location="cpu")
            missing, unexpected = self.unet.load_state_dict(
                motion_module_state_dict, strict=False)
            assert len(unexpected) == 0

        # update base model using dreambooth
        if self.unet is None:
            print("Please select a pretrained model path.")
            raise ValueError("No diffusion model is selected.")
        else:
            # change to safetensors later
            self.personalized_model = os.path.join(
                all_model_base, DreamBooth)
            base_model_state_dict = {}
            with safe_open(self.personalized_model, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)

            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=scheduler_dict[sampler](
                **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to(device)

        self.lcm_lora_path = os.path.join(
            all_model_base, LCM_LORA)
        pipeline.unet = convert_lcm_lora(copy.deepcopy(
            self.unet), self.lcm_lora_path, LCM_LORA_weight)
        pipeline.to(device)


        self.models['vae'] = self.vae
        self.models['text_encoder'] = self.text_encoder
        self.models['tokenizer'] = self.tokenizer
        self.models['unet'] = self.unet
        self.models['pipeline'] = pipeline
        self.models['inference_config'] = self.inference_config
        return (self.models,)

    # update_lora_model(self, lora_model_dropdown):
    # pass lora first
        # lora_model_path = os.path.join(
        #     self.personalized_model_dir, lora_model_dropdown)
        # self.lora_model_state_dict = {}
        # if lora_model_dropdown == "none":
        #     pass
        # else:
        #     with safe_open(lora_model_dropdown, framework="pt", device="cpu") as f:
        #         for key in f.keys():
        #             self.lora_model_state_dict[key] = f.get_tensor(key)
        # return gr.Dropdown.update()

            # self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
class AnimateLCM:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "AnimateLCM_models": ("AnimateLCM_MODEL",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "sample_steps" : ("INT", {
                    "default" : 4,
                    "min": 1,
                    "max": 25,
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
            },
            "optional": {
                "width" : ("INT", {
                    "default" : 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                "height" : ("INT", {
                    "default" : 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                "video_length" : ("INT", {
                    "default" : 16,
                    "min": 12,
                    "max": 20,
                    "step": 1,
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                "cfg_scale" : ("FLOAT", {
                    "default" : 1,
                    "min": 1,
                    "max": 2,
                    "step": 0.05,
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                "rand_seed": ("INT", {
                    "default": 88,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "slider"
                })
            }
        }


    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    FUNCTION = "generate"
    #OUTPUT_NODE = False
    CATEGORY = "dezi-ai"


    def generate(
            AnimateLCM_models,
            positive_prompt,
            negative_prompt,
            sample_steps,
            width=512,
            height=512,
            video_length=16,
            cfg_scale:float = 1.0,
            rand_seed: int = 88,
        ):
        """
        Generate a node instance.
        """
        if rand_seed != -1 and rand_seed != "":
            torch.manual_seed(int(rand_seed))
        else:
            torch.seed()
        seed = torch.initial_seed()
        pipeline = AnimateLCM_models['pipeline']
        sample = pipeline(
            positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=sample_steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            video_length=video_length,
        ).videos
        videos = rearrange(sample, "b c t h w -> t b c h w")
        return (videos,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AnimateLCMModelLoader": AnimateLCMModelLoader,
    "AnimateLCM": AnimateLCM,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateLCMModelLoader": "AnimateLCMModelLoader",
    "AnimateLCM": "AnimateLCM",
}
