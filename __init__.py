from .nodes import SAM2PlusModelLoader, SAM2PlusVideoSegmentation

NODE_CLASS_MAPPINGS = {
    "SAM2PlusModelLoader": SAM2PlusModelLoader,
    "SAM2PlusVideoSegmentation": SAM2PlusVideoSegmentation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM2PlusModelLoader": "SAM2-Plus Model Loader",
    "SAM2PlusVideoSegmentation": "SAM2-Plus Video Segmentation"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]