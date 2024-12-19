import math
import torch

WIDTH_BASELINE = 800

def calculate_watermark_width(image_width, logo_scale_percentage):
    # Calculates the width of the watermark based on the image width and logo scale percentage.
    return math.ceil(image_width * logo_scale_percentage / 100)

def calculate_text_opacity(opacity):
        # Calculates text opacity based on input opacity to match watermark image
        return int((100 - opacity) * 255 / 100)

def adjust_font_size(image_width, font_size):
        # Adjusts font size to match bigger resolutions
        return int(image_width / WIDTH_BASELINE * font_size)

def generate_empty_image(width, height, batch_size=1):
        # Create a fully transparent image (RGBA with 0 alpha)
        r = torch.full([batch_size, height, width, 1], 0.0)
        g = torch.full([batch_size, height, width, 1], 0.0)
        b = torch.full([batch_size, height, width, 1], 0.0)
        a = torch.full([batch_size, height, width, 1], 0.0)

        # Concatenate all channels to form RGBA (4 channels)
        return (torch.cat((r, g, b, a), dim=-1))

def get_image_size(image):
        # Returns image's batch amount, height and width
        return (image.shape[0], image.shape[1], image.shape[2])