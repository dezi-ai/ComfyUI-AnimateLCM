import folder_paths
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "animatelcm"))

os.makedirs(os.path.join(folder_paths.models_dir, "AnimateLCM"), exist_ok=True)

folder_paths.add_model_folder_path("animatelcm", os.path.join(folder_paths.models_dir, "AnimateLCM"))
folder_paths.folder_names_and_paths['animatelcm'] = (folder_paths.folder_names_and_paths['animatelcm'][0], folder_paths.supported_pt_extensions) # | {'.json'})

animatelcm_checkpoints = folder_paths.get_filename_list("animatelcm")

assert len(animatelcm_checkpoints) > 0, "ERROR: No AnimateLCM checkpoints found. Please download & place them in the ComfyUI/models/AnimateLCM folder, and restart ComfyUI."

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]