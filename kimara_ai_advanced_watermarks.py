import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw, ImageFont
import comfy
import math
import numpy as np


MAX_RESOLUTION = 2048

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
                "font": ("STRING", {"default": "assets/fonts/DMSans-VariableFont_opsz,wght.ttf"}),
                "font_size": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "logo_scale_percentage": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "x_padding": ("INT", {"default": 69, "min": 0, "max": MAX_RESOLUTION, "step": 5}),
                "y_padding": ("INT", {"default": 69, "min": 0, "max": MAX_RESOLUTION, "step": 5}),
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
            image: A list of images or a single image to which the watermark will be applied.
            move_watermark: A boolean indicating whether to move the watermark across the image.
            logo_image: An optional image to be used as a watermark. If not provided, an empty image is used.
            mask: An optional mask to apply to the logo image for transparency effects.
            move_watermark_step: The number of pixels the watermark moves in each step (default: 10).
            watermark_text: The text to be used as the watermark (default: None). If not provided, no text watermark is applied.
            font: The font used for watermark text (default: None).
            font_size: The size of the watermark text font (default: 16).
            logo_scale_percentage: The percentage of the image width to scale the logo watermark (default: 25).
            x_padding: Horizontal padding for watermark positioning (default: 20).
            y_padding: Vertical padding for watermark positioning (default: 20).
            rotation: The degree of rotation for the watermark (default: 0).
            opacity: The opacity of the watermark (both image and text), where 0 is fully visible and 100 is fully transparent (default: 40).

        Returns:
            A tuple containing the list of watermarked images (images,).
        
        Workflow:
            1. Handles fallbacks for missing inputs such as logo image and watermark text.
            2. Calculates the size of the watermark based on the image width and logo scale percentage.
            3. Resizes the logo image to fit the desired watermark width.
            4. Calculates the opacity and adjusts the font size based on the image width.
            5. Initializes watermark and text positions if not already set.
            6. Iterates through each image and applies the watermark, adjusting position and movement as needed.
            7. Returns the list of watermarked images after applying both text and logo watermarks.
        """

        # Fallbacks for missing inputs
        logo_image = logo_image if logo_image is not None else self.generate_empty_image(1, 1)
        watermark_text = watermark_text if watermark_text is not None else ""

        # Get image and logo sizes
        image_count, image_height, image_width = self.get_image_size(image)
        watermark_width = self.calculate_watermark_width(image_width, logo_scale_percentage)
        resized_logo_image, resized_logo_width, resized_logo_height = self.resize_logo(logo_image, watermark_width)

        # Prepare text opacity and font size
        text_opacity = self.calculate_text_opacity(opacity)
        font_size = self.adjust_font_size(image_width, font_size)

        

        # Initialize watermark and text positions if not provided
        self.initialize_positions(image_width, image_height, resized_logo_width, resized_logo_height, font_size, x_padding, y_padding, move_watermark)

        # Watermarking loop
        x_direction, y_direction = self.x_direction, self.y_direction
        images = self.apply_watermark_to_images(image, move_watermark, resized_logo_image, resized_logo_width, resized_logo_height, watermark_text, font, font_size, text_opacity, opacity, move_watermark_step,
                                                logo_image, mask, x_padding, y_padding, image_width, image_height, x_direction, y_direction)

        # Update directions after the loop
        self.x_direction, self.y_direction = x_direction, y_direction

        return (images,)

    # Helper methods

    def calculate_watermark_width(self, image_width, logo_scale_percentage):
        # Calculates the width of the watermark based on the image width and logo scale percentage.
        return math.ceil(image_width * logo_scale_percentage / 100)

    def resize_logo(self, logo_image, watermark_width):
        # Resizes the logo image to match the specified watermark width while maintaining the logo's aspect ratio.

        logo_image_count, logo_image_height, logo_image_width = self.get_image_size(logo_image)
        resized_logo_image = self.resize_watermark_image(logo_image, logo_image_height, logo_image_width, watermark_width)
        resized_logo_count, resized_logo_height, resized_logo_width = self.get_image_size(resized_logo_image)
        return resized_logo_image, resized_logo_width, resized_logo_height

    def calculate_text_opacity(self, opacity):
        # Calculates text opacity based on input opacity to match watermark image
        return int((100 - opacity) * 255 / 100)

    def adjust_font_size(self, image_width, font_size):
        # Adjusts font size to match bigger resolutions
        width_baseline = 800
        return int(image_width / width_baseline * font_size)

    def initialize_positions(self, image_width, image_height, resized_logo_width, resized_logo_height, font_size, x_padding, y_padding, move_watermark):
        """
        Initializes the watermark and text positions based on the image dimensions and padding values.
        If the positions are not already set, they will be calculated to position the watermark and text
        in the bottom-right corner of the image, with optional padding.

        Args:
            image_width: The width of the base image.
            image_height: The height of the base image.
            resized_logo_width: The width of the resized logo.
            resized_logo_height: The height of the resized logo.
            font_size: The size of the text font for watermark text.
            x_padding: Horizontal padding to offset watermark and text from the image edges.
            y_padding: Vertical padding to offset watermark and text from the image edges.
            move_watermark: Whether to recalculate positions even if they are already set.
            
        Returns:
            None. This method updates the instance variables for watermark and text positions in-place.
        """

        print("1 WATERMARK_X & Y: ", self.watermark_x, self.watermark_y, "TEXT_X & Y: ", self.text_x, self.text_y)
        
        if self.watermark_x is None or self.watermark_y is None or not move_watermark:
            self.watermark_x = image_width - resized_logo_width - x_padding
            self.watermark_y = image_height - resized_logo_height - y_padding - font_size

        if self.text_x is None or self.text_y is None or not move_watermark:
            self.text_x = image_width - x_padding
            self.text_y = image_height - y_padding

        print("2 WATERMARK_X & Y: ", self.watermark_x, self.watermark_y, "TEXT_X & Y: ", self.text_x, self.text_y)

    def apply_watermark_to_images(self, image, move_watermark, resized_logo_image, resized_logo_width, resized_logo_height, watermark_text, font, font_size, text_opacity, opacity, move_watermark_step,
                                logo_image, mask, x_padding, y_padding, image_width, image_height, x_direction, y_direction):
        """
        Applies watermark text and a logo image to a batch of images, with optional movement of the watermark.

        Args:
            image: A PyTorch tensor or a batch of tensors representing images to be processed (shape: [N, C, H, W] or [C, H, W]).
            move_watermark: A boolean indicating whether the watermark should move across images.
            resized_logo_image: The resized logo image to be used as a watermark.
            resized_logo_width: The width of the resized logo.
            resized_logo_height: The height of the resized logo.
            watermark_text: Text to be used as a watermark.
            font: The font to be used for the watermark text.
            font_size: Size of the font used for watermark text.
            text_opacity: Opacity of the watermark text (0-255).
            opacity: Opacity of the logo image (0-100).
            move_watermark_step: Step size for moving the watermark if `move_watermark` is `True`.
            logo_image: The original logo image used for resizing.
            mask: An optional mask image for the logo.
            x_padding: Horizontal padding for positioning the watermark and text.
            y_padding: Vertical padding for positioning the watermark and text.
            image_width: The width of the base image.
            image_height: The height of the base image.
            x_direction: The current horizontal direction for moving the watermark.
            y_direction: The current vertical direction for moving the watermark.

        Returns:
            A list of watermarked images, each with the applied watermark text and logo.
        """
        
        images = []
        for idx, image in enumerate(image):
            if move_watermark:
                self.watermark_x, self.watermark_y, self.text_x, self.text_y, x_direction, y_direction, self.rotation = \
                    self.calculate_watermark_position(
                        self.watermark_x, self.watermark_y, resized_logo_width, resized_logo_height, 
                        self.text_x, self.text_y, 0, font_size, image_width, image_height, 
                        move_watermark_step, x_direction, y_direction, x_padding, y_padding, self.rotation
                    )

            # Apply text and logo to the image
            image, text_width = self.draw_watermark_text(image, watermark_text, font_size, self.text_x, self.text_y, font, text_opacity)
            watermarked_image = self.add_logo_image(
                image, resized_logo_image, self.watermark_x, self.watermark_y, opacity, self.rotation, mask
            )

            images.append(watermarked_image)
        
        return images

    def generate_empty_image(self, width, height, batch_size=1):
        # Create a fully transparent image (RGBA with 0 alpha)
        r = torch.full([batch_size, height, width, 1], 0.0)
        g = torch.full([batch_size, height, width, 1], 0.0)
        b = torch.full([batch_size, height, width, 1], 0.0)
        a = torch.full([batch_size, height, width, 1], 0.0)

        # Concatenate all channels to form RGBA (4 channels)
        return (torch.cat((r, g, b, a), dim=-1))

    def get_image_size(self, image):
        # Returns image's batch amount, height and width
        return (image.shape[0], image.shape[1], image.shape[2])

    def calculate_watermark_position (self, watermark_x, watermark_y, resized_logo_width, resized_logo_height, text_x, text_y, text_width, font_size, image_width, image_height, move_watermark_step, x_direction, y_direction, x_padding, y_padding, rotation):
        
        """
        Used only if move_watermark is True.
        Calculates suitable coordinates for the next watermark location, ensuring that the watermark stays within the image boundaries, especially in larger batch sizes.

        Args:
            watermark_x: The current X-coordinate for the watermark position.
            watermark_y: The current Y-coordinate for the watermark position.
            resized_logo_width: The width of the resized logo image.
            resized_logo_height: The height of the resized logo image.
            text_x: The current X-coordinate for the text position.
            text_y: The current Y-coordinate for the text position.
            text_width: The width of the drawn text.
            font_size: The font size used for the text.
            image_width: The width of the base image.
            image_height: The height of the base image.
            move_watermark_step: The step size for moving the watermark and text.
            x_direction: The current X-direction (1 for right, -1 for left).
            y_direction: The current Y-direction (1 for down, -1 for up).
            x_padding: Horizontal offset to ensure watermark stays within bounds.
            y_padding: Vertical offset to ensure watermark stays within bounds.
            rotation: The current rotation applied to the watermark, to adjust position accordingly.

        Returns:
            A tuple containing the updated coordinates for the watermark and text (next_wm_x, next_wm_y, next_text_x, next_text_y), 
            as well as the updated X and Y directions and the new rotation value. These values are used for the next iteration of watermark placement.
        """

        # Define image padded borders
        image_top = 0 + y_padding
        image_left = 0 + x_padding
        image_bottom = image_height - y_padding
        image_right = image_width - x_padding

        # Calculate next positions based on the current direction and step
        next_wm_x = watermark_x + x_direction * move_watermark_step
        next_wm_y = watermark_y + y_direction * move_watermark_step
        next_text_x = text_x + x_direction * move_watermark_step
        next_text_y = text_y + y_direction * move_watermark_step

        # Define watermark and text bounding box coordinates
        wm_left = next_wm_x
        wm_right = next_wm_x + resized_logo_width
        wm_top = next_wm_y
        wm_bottom = next_wm_y + resized_logo_height

        text_left = next_text_x - text_width
        text_right = next_text_x
        text_top = next_text_y - font_size
        text_bottom = next_text_y

        # Check for collisions and adjust position accordingly
        if min(wm_left, text_left) < image_left:
            # Collision on the left side, reverse X direction
            overlap_x = image_left - min(wm_left, text_left)
            next_wm_x += (move_watermark_step - overlap_x) * -x_direction
            next_text_x += (move_watermark_step - overlap_x) * -x_direction
            x_direction = 1

        if max(wm_right, text_right) > image_right:
            # Collision on the right side, reverse X direction
            overlap_x = max(wm_right, text_right) - image_right
            next_wm_x += (move_watermark_step - overlap_x) * -x_direction
            next_text_x += (move_watermark_step - overlap_x) * -x_direction
            x_direction = -1

        if min(wm_top, text_top) < image_top:
            # Collision on the top, reverse Y direction
            overlap_y = image_top - min(wm_top, text_top)
            next_wm_y += (move_watermark_step - overlap_y) * -y_direction
            next_text_y += (move_watermark_step - overlap_y) * -y_direction
            y_direction = 1

        if max(wm_bottom, text_bottom) > image_bottom:
            # Collision on the bottom, reverse Y direction
            overlap_y = max(wm_bottom, text_bottom) - image_bottom
            next_wm_y += (move_watermark_step - overlap_y) * -y_direction
            next_text_y += (move_watermark_step - overlap_y) * -y_direction
            y_direction = -1

        # Spin that shi- logo if rotation is applied
        if rotation > 0:
            rotation += rotation

        # Returns integer values for next position, directions and amount of rotation for next loop
        return int(next_wm_x), int(next_wm_y), int(next_text_x), int(next_text_y), x_direction, y_direction, rotation

    def resize_watermark_image(self, logo_image, original_logo_height, original_logo_width, logo_width):

        """
        Resizes a watermark image to fit within specified dimensions while preserving aspect ratio.

        Args:
            logo_image: The image tensor to be resized.
            original_logo_height: Original height of the logo image.
            original_logo_width: Original width of the logo image.
            logo_width: Desired logo width. If 0, defaults to MAX_RESOLUTION or original dimensions.

        Returns:
            A resized logo image tensor.
        """

        if logo_width <= 0:
            logo_width = MAX_RESOLUTION if original_logo_height < MAX_RESOLUTION else original_logo_width

        # Calculate resize ratio
        ratio = min(logo_width / original_logo_width, MAX_RESOLUTION / original_logo_height)
        new_width = round(original_logo_width * ratio)
        new_height = round(original_logo_height * ratio)

        # Resize the logo image
        resized_logo_image = logo_image.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        resized_logo_image = F.interpolate(resized_logo_image, size=(new_height, new_width), mode="nearest")
        resized_logo_image = resized_logo_image.permute(0, 2, 3, 1)  # Change back to (N, H, W, C)
        resized_logo_image = torch.clamp(resized_logo_image, 0, 1)  # Ensure values are within [0, 1]

        return resized_logo_image

    def draw_watermark_text(self, image_tensor, text, font_size, pos_x, pos_y, font_path, text_opacity):

        """
        Draws semi-transparent text on an image.

        Args:
            image_tensor: A PyTorch tensor representing the image (shape: [C, H, W]).
            text: The text to be drawn.
            font_size: Size of the text font.
            pos_x: X-coordinate for the text position.
            pos_y: Y-coordinate for the text position.
            font_path: Path to the font file to use.
            text_opacity: Opacity of the text (0-255).

        Returns:
            A tuple containing the updated image tensor and the width of the drawn text.
        """

        # Convert the PyTorch tensor to a NumPy array and scale to uint8
        image_pil = Image.fromarray(np.clip(image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).convert('RGBA')

        # Create a transparent layer for the text
        transparent_layer = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))

        # Set text color with opacity
        text_color = (255, 255, 255, text_opacity)

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
        pos_x = max(0, pos_x - text_width)
        pos_y = max(0, pos_y - font_size)

        # Draw the text on the image
        draw.text((pos_x, pos_y), text, fill=text_color, font=font)

        # Merge the image and text layer
        image_pil_with_text = Image.alpha_composite(image_pil, transparent_layer).convert('RGB')

        # Convert the result back to tensor and return with text width
        image_tensor = torch.from_numpy(np.array(image_pil_with_text).astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor, text_width
        
    def add_logo_image(self, image_tensor, logo_image_tensor, watermark_x, watermark_y, opacity, rotation, mask=None):

        """
        Applies a logo image as a watermark to a image, with optional rotation, opacity, and masking.

        Args:
            image_tensor: A PyTorch tensor representing the image that's receiving the watermark (shape: [C, H, W]).
            logo_image_tensor: A PyTorch tensor representing the logo image (shape: [C, H, W]).
            watermark_x: The X-coordinate for the position of the watermark (logo).
            watermark_y: The Y-coordinate for the position of the watermark (logo).
            opacity: The opacity of the logo (0 to 100, where 100 is fully visible and 0 is fully transparent).
            rotation: The angle (in degrees) to rotate the logo image.
            mask (optional): A PyTorch tensor representing a mask to apply to the logo image (shape: [C, H, W]). The mask is applied as an alpha channel to the logo.

        Returns:
            A PyTorch tensor representing the image with the applied logo watermark (shape: [C, H, W]).
        """

        # Convert image and logo image to PIL
        image_pil = Image.fromarray(np.clip(image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8))
        logo_image_pil = Image.fromarray(np.clip(logo_image_tensor.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).convert('RGBA')

        # Rotate the logo image
        logo_image_pil = logo_image_pil.rotate(rotation, expand=True)

        # Apply the mask (if provided)
        if mask is not None:
            mask_pil = Image.fromarray(np.clip(mask.cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)).resize(logo_image_pil.size)
            logo_image_pil.putalpha(ImageOps.invert(mask_pil))

        _, _, _, alpha = logo_image_pil.split()
        alpha = alpha.point(lambda x: int(x * (1 - opacity / 100)))
        logo_image_pil.putalpha(alpha)

        # Paste the logo onto the image
        image_pil.paste(logo_image_pil, (watermark_x, watermark_y), logo_image_pil)

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