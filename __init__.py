"""
ComfyUI Custom Nodes for Kimara.ai
"""

try:
    from .kimara_ai_advanced_watermarks import KimaraAIWatermarker, KimaraAIBatchImages
except ImportError as e:
    print(f"Error importing nodes: {e}")

# Maps internal node names to their class implementations
NODE_CLASS_MAPPINGS = {
    "KimaraAIWatermarker": KimaraAIWatermarker,
    "KimaraAIBatchImages": KimaraAIBatchImages
}

# Maps internal node names to their display names in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "KimaraAIWatermarker": "Kimara.ai Advanced Watermarker",
    "KimaraAIBatchImages": "Kimara.ai Batch Images"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
