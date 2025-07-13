from .nodes import NODE_CLASS_MAPPINGS as comfyui_smooth_step_lora_loader_nodes, NODE_DISPLAY_NAME_MAPPINGS as comfyui_smooth_step_lora_loader_nodes_display
from .experimental_nodes import NODE_CLASS_MAPPINGS as comfyui_smooth_step_lora_loader_experimental_nodes, NODE_DISPLAY_NAME_MAPPINGS as comfyui_smooth_step_lora_loader_experimental_nodes_display

NODE_CLASS_MAPPINGS = {
    **comfyui_smooth_step_lora_loader_nodes,
    **comfyui_smooth_step_lora_loader_experimental_nodes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **comfyui_smooth_step_lora_loader_nodes_display,
    **comfyui_smooth_step_lora_loader_experimental_nodes_display,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]