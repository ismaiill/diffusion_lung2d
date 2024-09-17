
Run the script `process_data_to_1chanelPNG.py` to convert the images from the dataset into `.png` images (grayscale). You need to create a directory called `datasets/lung_8_256/` beforehand:

```bash
mkdir -p datasets/lung_8_256/
python process_data_to_1chanelPNG.py

Model Training
To run training:

Set MODEL_FLAGS. For instance:

export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"
