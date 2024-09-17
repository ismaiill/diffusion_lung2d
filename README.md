## Image Processing and Training Instructions

## Preprocessing

Run the script `process_data_to_1chanelPNG.py` to convert the images from the dataset into .png images (grayscale). You need to create a directory called `datasets/lung_8_256/` beforehand.

## Training Setup

To run training, follow these steps:

1. Set `MODEL_FLAGS`. For instance:
   ```bash
   export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"
   ```

2. Set `DIFFUSION_FLAGS`. For instance:
   ```bash
   export DIFFUSION_FLAGS="--diffusion_steps 7000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
   ```

3. Set `TRAIN_FLAGS`. For instance:
   ```bash
   export TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
   ```

## Start Training

To start training, run the command:

bash
python image_train.py --data_dir PATH_TO_DATASET $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

To specify the location where checkpoints will be saved:

bash
export OPENAI_LOGDIR=".../diffusion_lung_2d/models"



## Sampling

To sample, run the command:

bash
python image_sample.py --model_path .../diffusion_lung_2d/models/model010000.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples NUMBER_OF_SAMPLES


To specify the location where samples will be saved:

bash
export OPENAI_LOGDIR=".../diffusion_lung_2d/results"
