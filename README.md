### Image Conversion

Run the script `process_data_to_1chanelPNG.py` to convert the dataset images into `.png` format (grayscale). Ensure you create a directory named `datasets/lung_8_256/` beforehand.

### Training Setup

1. **Set `MODEL_FLAGS`**  
   Example:  
   ```bash
   export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"
