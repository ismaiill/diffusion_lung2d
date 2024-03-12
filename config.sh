#!/bin/bash

export PYTHONPATH=~/diffusion-lung:$PYTHONPATH
export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"
export DIFFUSION_FLAGS="--diffusion_steps 7000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
