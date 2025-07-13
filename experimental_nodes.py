import torch
import comfy.utils
import copy

# We will target key layers in the UNet's diffusion model.
# This avoids altering the VAE or CLIP model, focusing on visual generation.
TARGET_LAYER_KEYWORDS = [
    "diffusion_model.input_blocks",
    "diffusion_model.middle_block",
    "diffusion_model.output_blocks"
]

# More specific keywords if we want to be even more targeted, but the above is a good start.
# TARGET_LAYER_KEYWORDS = ["attention", "res", "ff"] 

class CheckpointSmoothStepAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "effect_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_smooth_step"
    CATEGORY = "model_experimental"

    def smooth_step_function(self, x):
        x = x.to(torch.float32)
        return 3 * x*x - 2 * x*x*x

    def apply_smooth_step(self, model, strength, effect_scale):
        if strength == 0 and effect_scale == 1.0:
            return (model,)

        new_model = model.clone()
        new_model.model = copy.deepcopy(model.model)
        state_dict = new_model.model.state_dict()
        
        keys_to_process = [key for key in state_dict if any(keyword in key for keyword in TARGET_LAYER_KEYWORDS)]
        
        if not keys_to_process:
            print("Warning: No target layers found for Checkpoint Smooth Step. Check TARGET_LAYER_KEYWORDS. Passing model through without changes.")
            return (new_model,)

        pbar = comfy.utils.ProgressBar(len(keys_to_process))
        print(f"Applying Checkpoint Smooth Step to {len(keys_to_process)} layers...")

        for key in keys_to_process:
            if state_dict[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                tensor = state_dict[key].clone() # Work on a copy
                original_dtype = tensor.dtype
                
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)

                if min_val < max_val:
                    # --- Smooth Step Application ---
                    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-7)
                    adjusted_tensor = self.smooth_step_function(normalized_tensor)
                    denormalized_tensor = min_val + adjusted_tensor * (max_val - min_val)
                    
                    # --- Mix with original tensor based on 'strength' ---
                    # This creates the initial "direction" of the change
                    mixed_tensor = tensor * (1 - strength) + denormalized_tensor * strength
                    
                    # --- Scale the final effect ---
                    # Calculate the change (delta) from the original tensor
                    delta = mixed_tensor - tensor
                    
                    # Apply the effect_scale to the delta and add it back to the original
                    final_tensor = tensor + delta * effect_scale
                    
                    state_dict[key].copy_(final_tensor.to(original_dtype))
            
            pbar.update(1)

        print("Finished applying Checkpoint Smooth Step.")
        return (new_model,)

# The __init__.py should handle the mappings. We just define the classes here.
NODE_CLASS_MAPPINGS = {
    "CheckpointSmoothStep": CheckpointSmoothStepAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSmoothStep": "Checkpoint Smooth Step (Adv)"
}