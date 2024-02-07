import hashlib
from pathlib import Path
from typing import Callable, Union
from collections.abc import Iterable
from time import time

import numpy as np

import folder_paths
from comfy.model_base import SD21UNCLIP, SDXL, BaseModel, SDXLRefiner, SVD_img2vid, model_sampling
from comfy.model_management import xformers_enabled
from comfy.model_patcher import ModelPatcher

import comfy.model_sampling
import comfy_extras.nodes_model_advanced


class IsChangedHelper:
    def __init__(self):
        self.val = 0
    
    def no_change(self):
        return self.val
    
    def change(self):
        self.val = (self.val + 1) % 100


class ModelSamplingConfig:
    def __init__(self, beta_schedule: str, linear_start: float=None, linear_end: float=None):
        self.sampling_settings = {"beta_schedule": beta_schedule}
        if linear_start is not None:
            self.sampling_settings["linear_start"] = linear_start
        if linear_end is not None:
            self.sampling_settings["linear_end"] = linear_end
        self.beta_schedule = beta_schedule  # keeping this for backwards compatibility


class ModelSamplingType:
    EPS = "eps"
    V_PREDICTION = "v_prediction"
    LCM = "lcm"


def factory_model_sampling_discrete_distilled(original_timesteps=50):
    class ModelSamplingDiscreteDistilledEvolved(comfy_extras.nodes_model_advanced.ModelSamplingDiscreteDistilled):
        def __init__(self, *args, **kwargs):
            self.original_timesteps = original_timesteps  # normal LCM has 50
            super().__init__(*args, **kwargs)
    return ModelSamplingDiscreteDistilledEvolved


# based on code in comfy_extras/nodes_model_advanced.py
def evolved_model_sampling(model_config: ModelSamplingConfig, model_type: int, alias: str):
    # if LCM, need to handle manually
    if BetaSchedules.is_lcm(alias):
        sampling_type = comfy_extras.nodes_model_advanced.LCM
        if alias == BetaSchedules.LCM_100:
            sampling_base = factory_model_sampling_discrete_distilled(original_timesteps=100)
        elif alias == BetaSchedules.LCM_25:
            sampling_base = factory_model_sampling_discrete_distilled(original_timesteps=25)
        else:
            sampling_base = comfy_extras.nodes_model_advanced.ModelSamplingDiscreteDistilled
        class ModelSamplingAdvancedEvolved(sampling_base, sampling_type):
            pass
        # NOTE: if I want to support zsnr, this is where I would add that code
        return ModelSamplingAdvancedEvolved(model_config)
    # otherwise, use vanilla model_sampling function
    return model_sampling(model_config, model_type)


class BetaSchedules:
    AUTOSELECT = "autoselect"
    SQRT_LINEAR = "sqrt_linear (AnimateDiff)"
    LINEAR_ADXL = "linear (AnimateDiff-SDXL)"
    LINEAR = "linear (HotshotXL/default)"
    LCM = "lcm"
    LCM_100 = "lcm[100_ots]"
    LCM_25 = "lcm[25_ots]"
    LCM_SQRT_LINEAR = "lcm >> sqrt_linear"
    USE_EXISTING = "use existing"
    SQRT = "sqrt"
    COSINE = "cosine"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"

    LCM_LIST = [LCM, LCM_100, LCM_25, LCM_SQRT_LINEAR]

    ALIAS_LIST = [AUTOSELECT, USE_EXISTING, SQRT_LINEAR, LINEAR_ADXL, LINEAR, LCM, LCM_100, LCM_SQRT_LINEAR, # LCM_25 is purposely omitted
                  SQRT, COSINE, SQUAREDCOS_CAP_V2]

    ALIAS_MAP = {
        SQRT_LINEAR: "sqrt_linear",
        LINEAR_ADXL: "linear", # also linear, but has different linear_end (0.020)
        LINEAR: "linear",
        LCM_100: "linear",  # distilled, 100 original timesteps
        LCM_25: "linear",  # distilled, 25 original timesteps
        LCM: "linear",  # distilled
        LCM_SQRT_LINEAR: "sqrt_linear", # distilled, sqrt_linear
        SQRT: "sqrt",
        COSINE: "cosine",
        SQUAREDCOS_CAP_V2: "squaredcos_cap_v2",
    }

    @classmethod
    def is_lcm(cls, alias: str):
        return alias in cls.LCM_LIST

    @classmethod
    def to_name(cls, alias: str):
        return cls.ALIAS_MAP[alias]
    
    @classmethod
    def to_config(cls, alias: str) -> ModelSamplingConfig:
        linear_start = None
        linear_end = None
        if alias == cls.LINEAR_ADXL:
            # uses linear_end=0.020
            linear_end = 0.020
        return ModelSamplingConfig(cls.to_name(alias), linear_start=linear_start, linear_end=linear_end)
    
    @classmethod
    def to_model_sampling(cls, alias: str, model: ModelPatcher):
        if alias == cls.USE_EXISTING:
            return None
        ms_obj = evolved_model_sampling(cls.to_config(alias), model_type=model.model.model_type, alias=alias)
        return ms_obj

    @staticmethod
    def get_alias_list_with_first_element(first_element: str):
        new_list = BetaSchedules.ALIAS_LIST.copy()
        element_index = new_list.index(first_element)
        new_list[0], new_list[element_index] = new_list[element_index], new_list[0]
        return new_list


class BetaScheduleCache:
    def __init__(self, model: ModelPatcher): 
        self.model_sampling = model.model.model_sampling

    def use_cached_beta_schedule_and_clean(self, model: ModelPatcher):
        model.model.model_sampling = self.model_sampling
        self.clean()

    def clean(self):
        self.model_sampling = None


class Folders:
    ANIMATEDIFF_MODELS = "animatediff_models"
    MOTION_LORA = "animatediff_motion_lora"
    VIDEO_FORMATS = "animatediff_video_formats"


def add_extension_to_folder_path(folder_name: str, extensions: Union[str, list[str]]):
    if folder_name in folder_paths.folder_names_and_paths:
        if isinstance(extensions, str):
            folder_paths.folder_names_and_paths[folder_name][1].add(extensions)
        elif isinstance(extensions, Iterable):
            for ext in extensions:
                folder_paths.folder_names_and_paths[folder_name][1].add(ext) 


def try_mkdir(full_path: str):
    try:
        Path(full_path).mkdir()
    except Exception:
        pass


# register motion models folder(s)
folder_paths.add_model_folder_path(Folders.ANIMATEDIFF_MODELS, str(Path(__file__).parent.parent / "models"))
folder_paths.add_model_folder_path(Folders.ANIMATEDIFF_MODELS, str(Path(folder_paths.models_dir) / Folders.ANIMATEDIFF_MODELS))
add_extension_to_folder_path(Folders.ANIMATEDIFF_MODELS, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.ANIMATEDIFF_MODELS))

# register motion LoRA folder(s)
folder_paths.add_model_folder_path(Folders.MOTION_LORA, str(Path(__file__).parent.parent / "motion_lora"))
folder_paths.add_model_folder_path(Folders.MOTION_LORA, str(Path(folder_paths.models_dir) / Folders.MOTION_LORA))
add_extension_to_folder_path(Folders.MOTION_LORA, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.MOTION_LORA))

# register video_formats folder
folder_paths.add_model_folder_path(Folders.VIDEO_FORMATS, str(Path(__file__).parent.parent / "video_formats"))
add_extension_to_folder_path(Folders.VIDEO_FORMATS, ".json")


def get_available_motion_models():
    return folder_paths.get_filename_list(Folders.ANIMATEDIFF_MODELS)


def get_motion_model_path(model_name: str):
    return folder_paths.get_full_path(Folders.ANIMATEDIFF_MODELS, model_name)


def get_available_motion_loras():
    return folder_paths.get_filename_list(Folders.MOTION_LORA)


def get_motion_lora_path(lora_name: str):
    return folder_paths.get_full_path(Folders.MOTION_LORA, lora_name)


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 50):
    h = hashlib.sha256()
    b = bytearray(1024*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()


def calculate_model_hash(model: ModelPatcher):
    unet = model.model.diff
    t = unet.input_blocks[1]
    m = hashlib.sha256()
    for buf in t.buffers():
        m.update(buf.cpu().numpy().view(np.uint8))
    return m.hexdigest()

BIGMIN = -(2**63-1)
BIGMAX = (2**63-1)

class ModelTypeSD:
    SD1_5 = "SD1.5"
    SD2_1 = "SD2.1"
    SDXL = "SDXL"
    SDXL_REFINER = "SDXL_Refiner"
    SVD = "SVD"


def get_sd_model_type(model: ModelPatcher) -> str:
    if model is None:
        return None
    elif type(model.model) == BaseModel:
        return ModelTypeSD.SD1_5
    elif type(model.model) == SDXL:
        return ModelTypeSD.SDXL
    elif type(model.model) == SD21UNCLIP:
        return ModelTypeSD.SD2_1
    elif type(model.model) == SDXLRefiner:
        return ModelTypeSD.SDXL_REFINER
    elif type(model.model) == SVD_img2vid:
        return ModelTypeSD.SVD
    else:
        return str(type(model.model).__name__)

def is_checkpoint_sd1_5(model: ModelPatcher):
    return False if model is None else type(model.model) == BaseModel

def is_checkpoint_sdxl(model: ModelPatcher):
    return False if model is None else type(model.model) == SDXL


def raise_if_not_checkpoint_sd1_5(model: ModelPatcher):
    if not is_checkpoint_sd1_5(model):
        raise ValueError(f"For AnimateDiff, SD Checkpoint (model) is expected to be SD1.5-based (BaseModel), but was: {type(model.model).__name__}")


# TODO: remove this filth when xformers bug gets fixed in future xformers version
def wrap_function_to_inject_xformers_bug_info(function_to_wrap: Callable) -> Callable:
    if not xformers_enabled:
        return function_to_wrap
    else:
        def wrapped_function(*args, **kwargs):
            try:
                return function_to_wrap(*args, **kwargs)
            except RuntimeError as e:
                if str(e).startswith("CUDA error: invalid configuration argument"):
                    raise RuntimeError(f"An xformers bug was encountered in AnimateDiff - this is unexpected, \
                                       report this to Kosinkadink/ComfyUI-AnimateDiff-Evolved repo as an issue, \
                                       and a workaround for now is to run ComfyUI with the --disable-xformers argument.")
                raise
        return wrapped_function


class Timer(object):
    __slots__ = ("start_time", "end_time")

    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self) -> None:
        self.start_time = time()

    def update(self) -> None:
        self.start()

    def stop(self) -> float:
        self.end_time = time()
        return self.get_time_diff()

    def get_time_diff(self) -> float:
        return self.end_time - self.start_time

    def get_time_current(self) -> float:
        return time() - self.start_time


# TODO: possibly add configuration file in future when needed?
# # Load config settings
# ADE_DIR = Path(__file__).parent.parent
# ADE_CONFIG_FILE = ADE_DIR / "ade_config.json"

# class ADE_Settings:
#     USE_XFORMERS_IN_VERSATILE_ATTENTION = "use_xformers_in_VersatileAttention"

# # Create ADE config if not present
# ABS_CONFIG = {
#     ADE_Settings.USE_XFORMERS_IN_VERSATILE_ATTENTION: True
# }
# if not ADE_CONFIG_FILE.exists():
#     with ADE_CONFIG_FILE.open("w") as f:
#         json.dumps(ABS_CONFIG, indent=4)
# # otherwise, load it and use values
# else:
#     loaded_values: dict = None
#     with ADE_CONFIG_FILE.open("r") as f:
#         loaded_values = json.load(f)
#     if loaded_values is not None:
#         for key, value in loaded_values.items():
#             if key in ABS_CONFIG:
#                 ABS_CONFIG[key] = value
