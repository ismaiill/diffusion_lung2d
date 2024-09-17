
Run the script `process_data_to_1chanelPNG.py` to convert the images from the dataset into `.png` images (grayscale). You need to create a directory called `datasets/lung_8_256/` beforehand:

```bash
mkdir -p datasets/lung_8_256/
python process_data_to_1chanelPNG.py
```

Model Training

To run training:

Set MODEL_FLAGS. For instance:

export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"

Set DIFFUSION_FLAGS. For instance:

Set TRAIN_FLAGS. For instance:

export TRAIN_FLAGS="--lr 2e-5 --batch_size 128"

To start training, run the command:

python image_train.py --data_dir PATH_TO_DATASET $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

Use the following command to specify the location where checkpoints will be saved:

export OPENAI_LOGDIR=".../diffusion_lung_2d/models"

Sampling

To generate samples, run the command:

python image_sample.py --model_path .../diffusion_lung_2d/models/model010000.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples NUMBER_OF_SAMPLES

Use the following command to specify the location where samples will be saved:

export OPENAI_LOGDIR=".../diffusion_lung_2d/results"
