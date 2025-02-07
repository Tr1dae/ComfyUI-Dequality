import os
import torch
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from numpy.random import Generator

class Dequality:
    def __init__(self):
        pass

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dequality"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "jpeg_artifact_level": ("INT", {"default": 65, "min": 0, "max": 100, "step": 5}),
                "noise_level": ("INT", {"default": 8, "min": 0.0, "max": 100, "step": 1}),
                "adjust_brightness": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "adjust_color": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "adjust_contrast": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "seed": ("INT", { "default": 0, "min": -1125899906842624, "max": 1125899906842624 }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS():
        return True

    def convert_to_rgb(self, img):
        # Convert from RGBA to RGB if necessary
        if img.mode == 'RGBA':
            return img.convert('RGB')
        return img
    def add_realistic_noise(self, img, base_noise_level):
        base_noise_level = base_noise_level / 1000
        img_array = np.array(img)

        # Base noise: Gaussian noise
        noise = np.random.normal(0, 255 * base_noise_level, img_array.shape)
        
        # Noise variation: Some pixels have stronger noise
        noise_variation = np.random.normal(0, 255 * (base_noise_level * 2), img_array.shape)
        variation_mask = np.random.rand(*img_array.shape[:2]) > 0.95  # Mask for stronger noise
        noise[variation_mask] += noise_variation[variation_mask]

        # Applying the noise to the image
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img_array)

    def dequality(self, pixels, noise_level, jpeg_artifact_level, adjust_color, adjust_contrast, adjust_brightness, seed):
        rng = np.random.default_rng(seed=abs(seed))

        img = Image.fromarray(np.clip(255. * pixels.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

        img = self.convert_to_rgb(img)

        if adjust_brightness == 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(rng.uniform(0.9, 1.1))

        if adjust_color == 1:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(rng.uniform(0.9, 1.1))

        if adjust_contrast == 1:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(rng.uniform(0.9, 1.1))

        if noise_level > 0:
            img = self.add_realistic_noise(img, base_noise_level=noise_level)

        final = img
        if jpeg_artifact_level < 100:
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = os.path.join(tmp_dir, 'tmpfile')
                img.save(temp_path, 'JPEG', quality=jpeg_artifact_level)
                with Image.open(temp_path) as final_img:
                    pp = torch.from_numpy(np.array(final_img).astype(np.float32) / 255.0).unsqueeze(0)
                    return (pp,)
                
        pp = torch.from_numpy(np.array(final).astype(np.float32) / 255.0).unsqueeze(0)
        return (pp,)

NODE_CLASS_MAPPINGS = {
    "Dequality": Dequality
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dequality": "Dequality"
}