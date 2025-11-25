import os
import sys
import torch
import numpy as np
import folder_paths
from PIL import Image
import shutil
import tempfile
import requests
from tqdm import tqdm
import inspect

# --- 0. REGISTER THE SAM2 FOLDER ---
if "sam2" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("sam2", os.path.join(folder_paths.models_dir, "sam2"))

# --- CONFIGURATION ---
SAM2_PLUS_URL = "https://huggingface.co/MCG-NJU/SAM2-Plus/resolve/main/checkpoint_phase123.pt"
DEFAULT_MODEL_NAME = "checkpoint_phase123.pt"

# --- HELPER: AUTO-DOWNLOADER ---
def download_sam2_plus_model(target_path):
    print(f"[SAM2+] Model missing. Downloading to: {target_path}")
    try:
        response = requests.get(SAM2_PLUS_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading SAM2-Plus") as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print("[SAM2+] Download complete!")
        return True
    except Exception as e:
        print(f"[SAM2+] Download failed: {e}")
        return False

# --- HELPER: COMPATIBILITY PATCHER (CRITICAL FIX) ---
def apply_compatibility_patch():
    """
    Patches load_video_frames to ignore 'frame_names' if the installed version doesn't support it.
    This fixes the 'unexpected keyword argument' error.
    """
    try:
        import sam2.utils.misc
        original_load = sam2.utils.misc.load_video_frames
        
        # Check if the function accepts 'frame_names'
        sig = inspect.signature(original_load)
        if 'frame_names' not in sig.parameters:
            print("[SAM2+] 🛠️ APPLYING PATCH: Your installed SAM2 version is old. Patching load_video_frames...")
            
            def patched_load_video_frames(*args, **kwargs):
                # If frame_names is passed but not supported, remove it
                if 'frame_names' in kwargs:
                    kwargs.pop('frame_names')
                return original_load(*args, **kwargs)
            
            # Replace the function in memory
            sam2.utils.misc.load_video_frames = patched_load_video_frames
            print("[SAM2+] 🛠️ Patch applied successfully.")
    except Exception as e:
        print(f"[SAM2+] Warning: Could not apply compatibility patch: {e}")

# --- HELPER: SETUP PATHS ---
def setup_sam2_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(current_dir, "sam2_plus_repo")
    
    # 1. Check for Repository Folder
    if not os.path.exists(repo_path):
        raise RuntimeError(f"Missing folder: {repo_path}\nPlease ensure 'sam2_plus_repo' exists inside the node directory.")

    # 2. Add to Python Path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path) # Insert at 0 to prioritize

    # 3. Check for Critical Subfolders
    required_subs = ["sam2_plus", "configs", "training"]
    missing = [sub for sub in required_subs if not os.path.exists(os.path.join(repo_path, sub))]
    
    if missing:
        raise RuntimeError(
            f"CRITICAL ERROR: Missing subfolders in 'sam2_plus_repo': {missing}\n"
            f"You MUST copy the '{missing[0]}' folder from the SAM2-Plus ZIP file into: {repo_path}"
        )
    
    return repo_path

# --- NODE 1: MODEL LOADER ---
class SAM2PlusModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        try:
            models = folder_paths.get_filename_list("sam2")
        except:
            models = []
            
        if DEFAULT_MODEL_NAME not in models:
            models.append(DEFAULT_MODEL_NAME)
            
        return {
            "required": {
                "model_name": (models,), 
            },
        }

    RETURN_TYPES = ("SAM2_MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM2-Plus"

    def load_model(self, model_name):
        repo_path = setup_sam2_paths()
        
        # 1. Resolve Model Path
        sam2_dir = os.path.join(folder_paths.models_dir, "sam2")
        ckpt_path = folder_paths.get_full_path("sam2", model_name)

        if not ckpt_path or not os.path.exists(ckpt_path):
            if model_name == DEFAULT_MODEL_NAME:
                expected_path = os.path.join(sam2_dir, DEFAULT_MODEL_NAME)
                os.makedirs(sam2_dir, exist_ok=True)
                if download_sam2_plus_model(expected_path):
                    ckpt_path = expected_path
                else:
                    raise FileNotFoundError(f"Failed to download {DEFAULT_MODEL_NAME}.")
            else:
                raise FileNotFoundError(f"Model '{model_name}' not found.")

        # 2. Resolve Config Path
        config_path = os.path.join(repo_path, "configs", "sam2.1", "sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml")
        if not os.path.exists(config_path):
             # Try nested search
             nested_config = os.path.join(repo_path, "sam2_plus", "configs", "sam2.1", "sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml")
             if os.path.exists(nested_config):
                 config_path = nested_config
             else:
                 raise FileNotFoundError(f"Config file missing at {config_path}")

        print(f"[SAM2+] Loaded Model: {ckpt_path}")
        return ({"ckpt_path": ckpt_path, "config_path": config_path},)

# --- NODE 2: SEGMENTATION ---
class SAM2PlusVideoSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "sam2_model": ("SAM2_MODEL",),
            },
            "optional": {
                "mask_hint": ("MASK",), 
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "segmented_images")
    FUNCTION = "segment"
    CATEGORY = "SAM2-Plus"

    def segment(self, images, sam2_model, mask_hint=None):
        setup_sam2_paths() # Ensure paths are correct before import
        apply_compatibility_patch() # FIX: Apply the patch before loading

        try:
            from sam2_plus.build_sam import build_sam2_video_predictor_plus
        except ImportError as e:
            raise RuntimeError(f"Import Error: {e}\nDid you copy the 'training' folder into 'sam2_plus_repo'?")

        print("[SAM2+] Building predictor...")
        try:
            predictor = build_sam2_video_predictor_plus(
                config_file=sam2_model["config_path"],
                ckpt_path=sam2_model["ckpt_path"],
                apply_postprocessing=False,
                hydra_overrides_extra=["++model.non_overlap_masks=false"],
                vos_optimized=False,
                task='mask'
            )
        except Exception as e:
            raise RuntimeError(f"Error building model: {e}")

        # Process Video
        temp_dir = tempfile.mkdtemp()
        try:
            for i in range(len(images)):
                img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(temp_dir, f"{i:05d}.jpg"))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=temp_dir)
                
                if mask_hint is not None:
                    masks = mask_hint if len(mask_hint.shape) == 3 else mask_hint.unsqueeze(0)
                    for i in range(masks.shape[0]):
                         m = (masks[i].cpu().numpy() > 0.5)
                         if m.max() > 0:
                            predictor.add_new_mask(state, 0, i+1, m)
                
                results = {}
                for idx, _, logits, _, _ in predictor.propagate_in_video(state):
                     combined = torch.zeros(logits.shape[1:], device="cpu")
                     if len(logits) > 0:
                         combined = (logits > 0.0).float().max(dim=0)[0].cpu()
                     results[idx] = combined

            H, W = images.shape[1], images.shape[2]
            out_tensor = torch.zeros((len(images), H, W))
            for i in range(len(images)):
                if i in results:
                    out_tensor[i] = results[i]

        finally:
            shutil.rmtree(temp_dir)

        return (out_tensor, images)