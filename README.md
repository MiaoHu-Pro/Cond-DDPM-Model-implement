# Conditional DDPM implementation

This folder implements a **class-conditional Denoising Diffusion Probabilistic
Model (Conditional DDPM)** for MNIST. Unlike an unconditional DDPM, the model
receives both the diffusion timestep $t$ and a requested digit label $y$. It
can therefore generate a selected digit, such as several new images of `7`.

## How it works

Training first adds Gaussian noise to a clean MNIST image $x_0$ at a randomly
sampled timestep:

$$
x_t=\sqrt{\bar{\alpha}_t}\,x_0
    +\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

The conditional U-Net predicts that noise from the noisy image, timestep, and
digit label:

$$
\hat{\epsilon}=\epsilon_\theta(x_t,t,y).
$$

It is trained with the standard DDPM noise-prediction loss:

$$
L=\mathbb{E}_{x_0,t,y,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t,y)\rVert_2^2\right].
$$

In the code, **a learned label embedding $y$ is added to the sinusoidal timestep
embedding**. This combined condition is injected into each U-Net convolution
block. During generation, the same label is supplied at every reverse
diffusion step, guiding random Gaussian noise toward the requested digit.

## Main files

- `code-1.py`: the original conditional DDPM demonstration.
- `cond-ddpm.py`: the corrected training and testing implementation. It
  supports checkpoints, mixed precision, gradient clipping, saved preview
  images, and reconstruction of the model configuration during testing.

The test workflow generates new images; it does not retrieve matching samples
from the MNIST test set.

## Train and generate a digit

```bash
cd ./cond-diffusion-model

# Train and save checkpoints/mnist-cond-diffusion.pt.
python cond-ddpm.py --mode train --device cuda --epochs 10

# Load the checkpoint and generate 16 images conditioned on digit 7.
python cond-ddpm.py \
    --mode test \
    --device cuda \
    --digit 7 \
    --num-samples 16
```

Training saves the `.pt` checkpoint, loss history, an epoch preview for all ten
digit classes, and `training-final-preview.png`. Test mode saves a grid such as
`checkpoints/test-digit-7.png`.
