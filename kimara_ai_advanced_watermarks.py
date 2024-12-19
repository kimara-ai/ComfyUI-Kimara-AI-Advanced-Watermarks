import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw, ImageFont
import comfy
import math
import numpy as np

from . import utils


class KimaraAIBatchImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),              
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Kimara.ai"

    def execute(self, images):
        """
        Concatenates a list of images into a single batch tensor along the first dimension.

        Args:
            images: A list of images (tensors) to be batched together.

        Returns:
            A batch image tensor created by concatenating the input images.
        """

        number_of_images = len(images)
        batch_image = images[0]

        if number_of_images > 1:
            
            for idx, next_image in enumerate(images, start=1):
                if idx == 1:
                    # Skip processing for the first image (already assigned to batch_image)
                    continue

                batch_image = torch.cat((batch_image, next_image), dim=0)
            
            return (batch_image,)

        

class KimaraAIWatermarker:
    
    def __init__(self):
        self.watermark_x = None
        self.watermark_y = None
        self.text_x = None
        self.text_y = None
        self.x_direction = -1
        self.y_direction = -1
        self.rotation = 0
        self.current_resolution = None
        self.previous_resolution = None  

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image": ("IMAGE",),
                "move_watermark": ("BOOLEAN", [True, False])                
            },
            "optional": {
                "logo_image": ("IMAGE",),
                "mask": ("MASK",),
                "move_watermark_step": ("INT", {"default": 10, "min": 1, "max": 500, "step": 1}),
                "watermark_text": ("STRING", {"multiline": False, "default": "Made by userxyz123", "lazy": True}),
                "font": ("STRING", {"default": "custom_nodes/ComfyUI-Kimara-AI-Advanced-Watermarks/assets/fonts/DMSans-VariableFont_opsz,wght.ttf"}),
                "font_size": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "logo_scale_percentage": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
                "x_padding": ("INT", {"default": 20, "min": 0, "max": 256, "step": 5}),
                "y_padding": ("INT", {"default": 20, "min": 0, "max": 256, "step": 5}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "opacity": ("FLOAT", {"default": 40, "min": 0, "max": 100, "step": 5})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "execute"
    CATEGORY = "Kimara.ai"

    def execute(self, image, move_watermark, logo_image=None, mask=None, move_watermark_step=10, watermark_text=None, font=None, font_size=16,
                logo_scale_percentage=25, x_padding=20, y_padding=20, rotation=0, opacity=40):

        """
        Applies watermark and logo overlays to an image (or a batch of images). 

        This method handles both watermark text and image logos, resizing and positioning them according to specified parameters.
        It also handles the movement of the watermark across the image, ensuring it remains within bounds.

        Args:
            :param image: A list of images or a single image to which the watermark will be applied.
            :param move_watermark: A boolean indicating whether to move the watermark across the image.
            :param logo_image: An optional image to be used as a watermark. If not provided, an empty image is used.
            :param mask: An optional mask to apply to the logo image for transparency effects.
            :param move_watermark_step: The number of pixels the watermark moves in each step (default: 10).
            :param watermark_text: The text to be used as the watermark (default: None). If not provided, no text watermark is applied.
            :param font: The font used for watermark text (default: None).
            :param font_size: The size of the watermark text font (default: 16).
            :param logo_scale_percentage: The percentage of the image width to scale the logo watermark (default: 25).
            :param x_padding: Horizontal padding for watermark positioning (default: 20).
            :param y_padding: Vertical padding for watermark positioning (default: 20).
            :param rotation: The degree of rotation for the watermark (default: 0).
            :param opacity: The opacity of the watermark (both image and text), where 0 is fully visible and 100 is fully transparent (default: 40).

        Returns:
            A tuple containing the list of watermarked images (images,).
        
        Workflow:
            1. Handles fallbacks for missing inputs such as logo image and watermark text.
            2. Calculates the size of the watermark based on the image width and logo scale percentage.
            3. Resizes the logo image to fit the desired watermark width.
            4. Adjusts the font size based on the image width.
            5. Initializes watermark and text positions if not already set.
            6. Iterates through each image and applies the watermark, adjusting position and movement as needed.
            7. Returns the list of watermarked images after applying both text and logo watermarks.
        """

        # Fallbacks for missing inputs
        logo_image = logo_image if logo_image is not None else utils.generate_empty_image(1, 1)
        watermark_text = watermark_text if watermark_text is not None else ""

        # Get image and logo sizes
        _, image_height, image_width = utils.get_image_size(image)
        self.current_resolution = (image_width, image_height)
        watermark_width = utils.calculate_watermark_width(image_width, logo_scale_percentage)

        # Resize logo image and adjust font size
        resized_logo_image, resized_logo_width, resized_logo_height = self.resize_watermark_image(logo_image, watermark_width)               
        font_size = utils.adjust_font_size(image_width, font_size)

        # Use the reset_position method to initialize values if None
        if not move_watermark or any(value is None for value in [self.watermark_x, self.watermark_y, self.text_x, self.text_y]):
            self.reset_position(image_width, image_height, resized_logo_width, resized_logo_height, font_size, x_padding, y_padding)
            
        # Watermarking loop
        images = self.apply_watermark_to_images(image, move_watermark, resized_logo_image, resized_logo_width, resized_logo_height, watermark_text, font, font_size, opacity, move_watermark_step,
                                                logo_image, mask, x_padding, y_padding, image_width, image_height)

        return (images,)

    def reset_position(self, image_width, image_height, resized_logo_width, resized_logo_height, font_size, x_padding, y_padding):
        # Reset (or initialize) positions of watermark and text
        self.watermark_x = image_width - resized_logo_width - x_padding
        self.watermark_y = image_height - resized_logo_height - y_padding - font_size
        self.text_x = image_width - x_padding
        self.text_y = image_height - y_padding

    def apply_watermark_to_images(self, image, move_watermark, resized_logo_image, resized_logo_width, resized_logo_height, watermark_text, font, font_size, opacity, move_watermark_step,
                                logo_image, mask, x_padding, y_padding, image_width, image_height):
        """
        Applies watermark text and a logo image to a batch of images, with optional movement of the watermark.

        Args:
            :param image: A PyTorch tensor or a batch of tensors representing images to be processed (shape: [N, C, H, W] or [C, H, W]).
            :param move_watermark: A boolean indicating whether the watermark should move across images.
            :param resized_logo_image: The resized logo image to be used as a watermark.
            :param resized_logo_width: The width of the resized logo.
            :param resized_logo_height: The height of the resized logo.
            :param watermark_text: Text to be used as a watermark.
            :param font: The font to be used for the watermark text.
            :param font_size: Size of the font used for watermark text.
            :param text_opacity: Opacity of the watermark text (0-255).
            :param opacity: Opacity of the logo image (0-100).
            :param move_watermark_step: Step size for moving the watermark if `move_watermark` is `True`.
            :param logo_image: The original logo image used for resizing.
            :param mask: An optional mask image for the logo.
            :param x_padding: Horizontal padding for positioning the watermark and text.
            :param y_padding: Vertical padding for positioning the watermark and text.
            :param image_width: The width of the base image.
            :param image_height: The height of the base image.

        Returns:
            A list of watermarked images, each with the applied watermark text and logo.
        """
        
        images = []
        for idx, image in enumerate(image):
            if move_watermark:
                self.calculate_watermark_position(resized_logo_width, resized_logo_height, 0, font_size, image_width, image_height, move_watermark_step, x_padding, y_padding)

            # Apply text and logo to the image
            image, text_width = self.draw_watermark_text(image, watermark_text, font_size, font, opacity)
            watermarked_image = self.add_logo_image(image, resized_logo_image, opacity, mask)

            images.append(watermarked_image)
        
        return images

    def calculate_watermark_position(self, resized_logo_width, resized_logo_height, text_width, font_size, image_width, image_height, move_watermark_step, x_padding, y_padding):
        """
        Calculates the position of the watermark and adjusts it based on movement and image boundaries.

        This method determines the next position of the watermark and text, handles collisions with image borders, and updates the positions accordingly.

        Args:
            :param resized_logo_width: The width of the resized logo image.
            :param resized_logo_height: The height of the resized logo image.
            :param text_width: The width of the watermark text.
            :param font_size: The size of the watermark text font.
            :param image_width: The width of the base image.
            :param image_height: The height of the base image.
            :param move_watermark_step: The number of pixels to move the watermark in each step.
            :param x_padding: Horizontal padding for watermark positioning.
            :param y_padding: Vertical padding for watermark positioning.

        Workflow:
            1. Checks if the resolution has changed; if so, resets watermark and text positions.
            2. Defines image borders to constrain watermark movement within bounds.
            3. Calculates the next positions for watermark and text based on current direction and step size.
            4. Adjusts the positions to prevent overlap with image borders, reversing direction if necessary.
        """

        if self.previous_resolution != self.current_resolution:
            self.reset_position(image_width, image_height, resized_logo_width, resized_logo_height, font_size, x_padding, y_padding)
            self.previous_resolution = self.current_resolution

        # Define image padded borders
        image_top = 0 + y_padding
        image_left = 0 + x_padding
        image_bottom = image_height - y_padding
        image_right = image_width - x_padding

        # Get next positions
        next_wm_x, next_wm_y, next_text_x, next_text_y = self.get_next_positions(move_watermark_step)

        # Check for collisions and adjust position
        self.watermark_x, self.text_x = self.handle_collisions(next_wm_x, resized_logo_width, next_text_x, -text_width, image_left, image_right, move_watermark_step, axis="x")
        self.watermark_y, self.text_y = self.handle_collisions(next_wm_y, resized_logo_height, next_text_y, -font_size, image_top, image_bottom, move_watermark_step, axis="y")

    def get_next_positions(self, move_watermark_step):
        # Calculate the next positions for the watermark and text
        x_step = self.x_direction * move_watermark_step
        y_step = self.y_direction * move_watermark_step

        next_wm_x = self.watermark_x + x_step
        next_wm_y = self.watermark_y + y_step
        next_text_x = self.text_x + x_step
        next_text_y = self.text_y + y_step 

        return next_wm_x, next_wm_y, next_text_x, next_text_y

    def handle_collisions(self, next_wm_pos, wm_width, next_text_pos, text_size, image_min, image_max, move_step, axis):
        # Handle collisions on both X and Y axis
        wm_min = next_wm_pos
        wm_max = next_wm_pos + wm_width
        text_min = next_text_pos + text_size
        text_max = next_text_pos

        # Determine direction attribute
        direction_attr = f"{axis}_direction"
        direction = getattr(self, direction_attr)

        if min(wm_min, text_min) < image_min:
            overlap = image_min - min(wm_min, text_min)
            next_wm_pos += (move_step - overlap) * -direction
            next_text_pos += (move_step - overlap) * -direction
            setattr(self, direction_attr, 1)

        if max(wm_max, text_max) > image_max:
            overlap = max(wm_max, text_max) - image_max
            next_wm_pos += (move_step - overlap) * -direction
            next_text_pos += (move_step - overlap) * -direction
            setattr(self, direction_attr, -1)

        return next_wm_pos, next_text_pos

    def resize_watermark_image(self, logo_image, logo_width):

        """
        Resizes a watermark image to fit within specified dimensions while preserving aspect ratio.

        Args:
            :param logo_image: The image tensor to be resized.
            :param original_logo_height: Original height of the logo image.
            :param original_logo_width: Original width of the logo image.
            :param logo_width: Desired logo width. If 0, defaults to current resolution of the base image or original dimensions.

        Returns:
            A resized logo image tensor.
        """

        image_width, image_height = self.current_resolution
        _, original_logo_height, original_logo_width = utils.get_image_size(logo_image)

        if logo_width <= 0:
            logo_width = image_width if original_logo_height < image_height else original_logo_width

        # Calculate resize ratio
        ratio = min(logo_width / original_logo_width, image_height / original_logo_height)
        new_width = round(original_logo_width * ratio)
        new_height = round(original_logo_height * ratio)

        # Resize the logo image
        resized_logo_image = logo_image.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        resized_logo_image = F.interpolate(resized_logo_image, size=(new_height, new_width), mode="nearest")
        resized_logo_image = resized_logo_image.permute(0, 2, 3, 1)  # Change back to (N, H, W, C)
        resized_logo_image = torch.clamp(resized_logo_image, 0, 1)  # Ensure values are within [0, 1]

        _, resized_logo_height, resized_logo_width = utils.get_image_size(resized_logo_image)

        return resized_logo_image, resized_logo_width, resized_logo_height

    def draw_watermark_text(self, image_tensor, text, font_size, font_path, opacity):

        """
        Draws semi-transparent text on an image.

        Args:
            :param image_tensor: A PyTorch tensor representing the image (shape: [C, H, W]).
            :param text: The text to be drawn.
            :param font_size: Size of the text font.
            :param font_path: Path to the font file to use.
            :param text_opacity: Opacity of the text (0-255).

        Returns:
            A tuple containing the updated image tensor and the width of the drawn text.
        """

        # Convert the PyTorch tensor to a NumPy array and scale to uint8
        image_pil = Image.fromarray(np.clip(image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).convert('RGBA')

        # Create a transparent layer for the text
        transparent_layer = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))

        # Set text color with opacity
        text_opacity = utils.calculate_text_opacity(opacity)
        text_color = (255, 255, 255, text_opacity) # White, maybe add custom colors as input later ?

        # Initialize drawing context
        draw = ImageDraw.Draw(transparent_layer)

        # Load the specified font or fall back to the default font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            print(f"Font '{font_path}' not found. Using the default font.")
            font = ImageFont.load_default()

        # Calculate text dimensions
        text_width = draw.textlength(text, font=font)

        # Adjust position to align text to the bottom-right corner
        pos_x = max(0, self.text_x - text_width)
        pos_y = max(0, self.text_y - font_size)

        # Draw the text on the image
        draw.text((pos_x, pos_y), text, fill=text_color, font=font)

        # Merge the image and text layer
        image_pil_with_text = Image.alpha_composite(image_pil, transparent_layer).convert('RGB')

        # Convert the result back to tensor and return with text width
        image_tensor = torch.from_numpy(np.array(image_pil_with_text).astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor, text_width
        
    def add_logo_image(self, image_tensor, logo_image_tensor, opacity, mask=None):

        """
        Applies a logo image as a watermark to a image, with optional rotation, opacity, and masking.

        Args:
            :param image_tensor: A PyTorch tensor representing the image that's receiving the watermark (shape: [C, H, W]).
            :param logo_image_tensor: A PyTorch tensor representing the logo image (shape: [C, H, W]).
            :param opacity: The opacity of the logo (0 to 100, where 100 is fully visible and 0 is fully transparent).
            :param mask (optional): A PyTorch tensor representing a mask to apply to the logo image (shape: [C, H, W]). The mask is applied as an alpha channel to the logo.

        Returns:
            A PyTorch tensor representing the image with the applied logo watermark (shape: [C, H, W]).
        """

        # Convert image and logo image to PIL
        image_pil = Image.fromarray(np.clip(image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8))
        logo_image_pil = Image.fromarray(np.clip(logo_image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).convert('RGBA')

        # Rotate the logo image
        logo_image_pil = logo_image_pil.rotate(self.rotation, expand=True)

        # Apply the mask (if provided)
        if mask is not None:
            mask_pil = Image.fromarray(np.clip(mask.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).resize(logo_image_pil.size)
            logo_image_pil.putalpha(ImageOps.invert(mask_pil))

        _, _, _, alpha = logo_image_pil.split()
        alpha = alpha.point(lambda x: int(x * (1 - opacity / 100)))
        logo_image_pil.putalpha(alpha)

        # Paste the logo onto the image
        image_pil.paste(logo_image_pil, (self.watermark_x, self.watermark_y), logo_image_pil)

        # Convert the result back to tensor and return
        image_tensor = torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor

NODE_CLASS_MAPPINGS = {
    "KimaraAIWatermarker": KimaraAIWatermarker,
    "KimaraAIBatchImages": KimaraAIBatchImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimaraAIWatermarker": "Kimara.ai Advanced Watermarker",
    "KimaraAIBatchImages": "Kimara.ai Batch Images"
}