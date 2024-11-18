# Kimara.ai's Advanced Watermarking Tools

**NOTE**: This custom node is still in development.

**NOTE**: This is just a quick ChatGPT assisted README-boilerplate.

The **KimaraAIWatermarker** custom node allows you to apply watermark text and logo overlays to images (or a batch of images). It provides features like customizable watermark movement, rotation, and opacity. You can also apply both text and logo watermarks simultaneously, with fine-tuned control over positioning and scaling.

## Features

- Add watermark text to images.
- Add logo watermarks to images.
- Move watermark across the image in steps.
- Rotate and adjust opacity of watermarks.
- Scale logo watermark based on image size.
- Batch processing for multiple images.

## Installation

Ensure you have the required dependencies installed. If you're using Python, you can install them via `requirements.txt`.

```
pip install -r requirements.txt
```

### Input Parameters

| Parameter               | Type               | Description                                                                                           |
| ----------------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `image`                 | `IMAGE`            | The image or list of images to which the watermark will be applied.                                   |
| `move_watermark`        | `BOOLEAN`          | Whether the watermark should move across the image.                                                   |
| `logo_image`            | `IMAGE` (optional) | The image to be used as the logo watermark. If not provided, an empty image will be used.             |
| `mask`                  | `MASK` (optional)  | The mask to apply to the logo image for transparency effects.                                         |
| `move_watermark_step`   | `INT`              | The number of pixels the watermark moves in each step (default: 10).                                  |
| `watermark_text`        | `STRING`           | The text to use as the watermark (default: None). If not provided, no text watermark will be applied. |
| `font`                  | `STRING`           | The font used for the watermark text (default: `assets/fonts/DMSans-VariableFont_opsz,wght.ttf`).     |
| `font_size`             | `INT`              | The font size for the watermark text (default: 16).                                                   |
| `logo_scale_percentage` | `INT`              | The percentage of the image width to scale the logo watermark (default: 25).                          |
| `x_padding`             | `INT`              | Horizontal padding for watermark positioning (default: 20).                                           |
| `y_padding`             | `INT`              | Vertical padding for watermark positioning (default: 20).                                             |
| `rotation`              | `INT`              | The rotation angle of the watermark (default: 0).                                                     |
| `opacity`               | `FLOAT`            | The opacity of the watermark (0 is fully visible, 100 is fully transparent).                          |

#### License

This project is licensed under the **GNU Affero General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.
