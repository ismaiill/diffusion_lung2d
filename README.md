Run the script process_data_to_1chanelPNG.py to convert the images from the dataset into .png images (grayscale). You need to create a directory called datasets/lung_8_256/ beforehand.

To run training, 

1 - Set MODELS_FLAGS. For instance, export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"

2 - Set DIFFUSION_FLAGS. For instance, export DIFFUSION_FLAGS="--diffusion_steps 7000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

3 - Set TRAIN_FLAGS. For instance, export TRAIN_FLAGS="--lr 2e-5 --batch_size 128"

To start trainng, run the command

python image_train.py --data_dir PATH TO DATASET  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

Use export OPENAI_LOGDIR=".../diffusion_lung_2d/models" to specify the location where checkpoints will be saved.

To sample, run the command:

python image_sample.py --model_path .../diffusion_lung_2d/models/model010000.pt  $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples NUMBER OF SAMPLES

Use export OPENAI_LOGDIR=".../diffusion_lung_2d/results" to specify the location where samples will be saved.
