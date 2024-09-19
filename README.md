# 2D Reconstruction of Ultrasound Lung Images via Diffusion

## Authors

- **Ismail Abouamal**  
  Caltech
  
- **Shi-Ning Sun**  
  Caltech  
## Abstract

We present an approach to reconstruct 2D images of lung organs from ultrasound scans using a diffusion model. As a first step, we start by training an unconditional model to generate 2D lung images from complete noise. Subsequently, we extend this methodology to develop a conditional model, utilizing ultrasound scans as measurements to guide the denoising process that generates 2D lung images.

## Introduction

### Background

Ultrasound imaging, also known as sonography, is an important technique used in diverse medical fields. It operates by emitting high-frequency sound waves into the body or surrounding environment, capturing their echoes as they bounce off obstacles and internal structures. These echoes are then processed by sensors to construct detailed images of the surrounding anatomy, including organs, blood vessels, and developing fetuses. In obstetrics, for example, ultrasound serves as a pivotal tool for assessing fetal health, monitoring development, and detecting potential abnormalities. Similarly, in cardiology, it offers a non-invasive means to evaluate the heart's structure and function with remarkable precision, sparing patients the invasiveness of surgical procedures.

Among its many applications, ultrasound finds a prominent role in abdominal imaging. By employing sonography to examine organs like the liver, clinicians can diagnose an array of conditions such as cysts and tumors. Notably, ultrasound presents several advantages over alternative imaging modalities such as X-rays or MRI. It offers a cost-effective alternative to MRI scans while eliminating the need for radiation exposure, making it a safer option for patients. Furthermore, it elevates the necessity of injecting contrast agents, ensuring a comfortable imaging experience for the patient.

Despite its versatility and efficacy, ultrasound encounters challenges in imaging certain organs, particularly the lungs. The technique's effectiveness is notably hindered by its limited ability to penetrate air-filled spaces. In lung imaging, this limitation becomes pronounced as ultrasound waves encounter difficulty traversing through the air-filled lung tissue essential for respiration. Consequently, visualizing deep structures beyond the lungs presents a challenge. The air within the lungs acts as a formidable barrier, impeding the ultrasound waves from reaching and visualizing deeper structures within the chest cavity.

Despite the challenges posed by the air-filled nature of the lungs, ultrasound continues to hold promise as a valuable diagnostic tool, driving ongoing research and advancements aimed at overcoming its limitations and expanding its utility in pulmonary care. The recent progress in machine learning techniques to solve inverse problems suggests a new avenue to overcome the challenges posed by imaging lung tissue with ultrasound. In this report, we discuss one possible approach that looks promising: using a diffusion model to obtain more accurate 2D lung images from ultrasound data.

![Speed of sound vs depth. The waves travel slower and dissipate at higher rate as they penetrate deeper into the chest.](chest.png)

### Research Problem

The importance of our research lies in the fact that accurately discerning the shape and structure of internal organs from ultrasound scans is crucial for healthcare practitioners. This capability makes it possible to conduct thorough visual inspections and detect early anomalies such as tumors, enabling timely intervention and treatment.

Our research proposes a potential solution to this problem. We aim to develop and implement the technique of diffusion and guided diffusion to augment the capabilities of ultrasound imaging, thereby enhancing its diagnostic utility. By leveraging this new technology, we seek to overcome the inherent limitations of traditional ultrasound methods. More precisely, our aim is to employ the recent progress in diffusion models to reconstruct the two-dimensional image of a lung organ from the ultrasound scans.

### Research Goal

Our project will consist of two steps. We will first train an unconditional diffusion model using our 2D lung images as the training data. Once we obtain this model and are able to generate new 2D lung images with this model, we will train a conditional model using the ultrasound measurements as guidance of the reverse diffusion process.

## Background

Diffusion models are a family of generative AI models inspired by non-equilibrium thermodynamics. Roughly speaking, the underlying structure consists of a Markov chain of diffusion steps where each step adds random noise to the data. 

Consider a data point \( \mathbf{x_0} \) sampled from an unknown distribution \( q(\mathbf{x}) \) and define the forward diffusion process as follows. Let \( T \in \mathbb{N} \) and \( \beta_1, \dots, \beta_T \) such that \( \beta_{t} \in (0,1) \) for \( 0 < t \leq N \) and the consider forward process defined by:

\[
\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \mathbf{\epsilon}_t, \quad \mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\]

We can check that 
\[
q(\bfx_t|\bfx_{t-1})= \gaussian(\sqrt{1 -\beta_t} \mathbf{x}_{t-1}, \beta_t).
\]

Using certain manipulations, we can find that 

\[
q(\bfx_{t-1} \mid \bfx_{t}, \bfx_{0})  \sim \mathcal{N}(\tilde{\mu}(\bfx_t, \bfx_0), \tilde{\beta}_t\mathbf{1} )
\]

with 

\[
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \bf\beta_t.
\]

## Experiments

We use 4500 2D lung images stored on the remote RTX machine. The training process uses an open-source diffusion model, which was also used in previous references. We use the parameters listed in Table 1 for the training process and generated 1000 2D lung images from the trained model.

| Diffusion steps | 4000 |
|------------------|------|
| Image size       | 256  |
| Number of channels| 96  |
| Number of residual blocks| 2 |
| Attention resolutions| 1   |
| Learning rate    | 0.0001 |
| Batch size       | 128  |

To confirm that the model does not overfit, we evaluated the degree of similarity between the generated images and the training images using the minimum \( L_2 \) distance.

In Fig. 1, we plot the distribution of all the minimum \( L_2 \) distances in each of the three categories. The average minimum distances are found to be similar across all three categories, confirming that our trained model is not biased toward a particular category.

![Distribution of minimum \( L_2 \) distance between each generated image and the training images.](dist.png)

## Discussion

There are some incomplete aspects of our study. We attempted using the Fréchet Inception Distance (FID) to evaluate the degree of similarity of the training set and the generated set. However, the FID we obtained are exponentially larger than what we would expect, indicating numerical issues in its computation. 

The original purpose of this study is to use RF data as guidance in the denoising process. Although we were not able to pursue this direction, it remains a topic of interest for further investigation.

## Code 

Run the script `process_data_to_1chanelPNG.py` to convert the images from the dataset into `.png` images (grayscale). You need to create a directory called `datasets/lung_8_256/` beforehand:

```bash
mkdir -p datasets/lung_8_256/
python process_data_to_1chanelPNG.py
```

## Model Training

To run training:

Set MODEL_FLAGS. For instance:

```bash
export MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 1 --attention_resolutions 1"
```

Set DIFFUSION_FLAGS. For instance:
```bash
export DIFFUSION_FLAGS="--diffusion_steps 7000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
```


Set TRAIN_FLAGS. For instance:
```bash

export TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
```

To start training, run the command:
```bash

python image_train.py --data_dir PATH_TO_DATASET $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Use the following command to specify the location where checkpoints will be saved:
```bash

export OPENAI_LOGDIR=".../diffusion_lung_2d/models"
```

## Sampling

To generate samples, run the command:
```bash

python image_sample.py --model_path .../diffusion_lung_2d/models/model010000.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples NUMBER_OF_SAMPLES
```

Use the following command to specify the location where samples will be saved:
```bash

export OPENAI_LOGDIR=".../diffusion_lung_2d/results"
```
