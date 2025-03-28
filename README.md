# :muscle: Stable Diffusion: 

 *This repository focuses on analyzing each component of Stable Diffusion.*

```
git pull https://github.com/mminh007/diffusers.git
```
## :rocket: Variant-AutoEncoder:

:link: <a href= "/VAE/README.md"> Explained of VAE  </a>
Variational autoencoders (VAE) are deep learning models composed of an encoder that learns to isolate the important latent variables from training data and a decoder that then uses those latent variables to reconstruct the input data.

However, whereas most autoencoder architectures encode a discrete, fixed representation of latent variables, VAEs encode a continuous, probabilistic representation of that latent space. This enables a VAE to not only accurately reconstruct the exact original input, but also use variational inference to generate new data samples that resemble the original input data.

VAEs are a subset of the larger category of autoencoders, a neural network architecture typically used in deep learning for tasks such as data compression, image denoising, anomaly detection and facial recognition.

:muscle: **Strongness:**

:white_check_mark: **Probabilistic Framework** – Unlike traditional autoencoders, VAE learns a probabilistic distribution rather than a fixed mapping, improving robustness and diversity in generated samples.

:white_check_mark: **Continuous and Structured Latent Space** – VAE enforces a smooth and continuous latent space, making it well-suited for generating coherent and meaningful samples.

:white_check_mark: **Scalability** – VAEs are relatively efficient and scalable to high-dimensional data, such as images and text.

:white_check_mark: **Uncertainty Estimation** – Since VAEs are probabilistic models, they provide uncertainty estimates, which are useful in applications like anomaly detection.

:-1: **Weakeness:** 

:white_check_mark: One of the biggest weaknesses is **the low reconstruction quality, as generated images often appear blurry**. This arises from the trade-off between the two main components in the loss function: **Reconstruction Loss** and **KL Divergence**. These two components often oppose each other, leading to a trade-off that results in reconstructions that are not as sharp as those produced by other models like GANs.

:white_check_mark: **Limited Expressiveness of Latent Space** – The assumption that the latent variables follow a Gaussian distribution can sometimes restrict the model’s ability to capture complex structures in the data.

:white_check_mark: **Mode Averaging** – Unlike GANs, which can model sharper distributions, VAEs tend to average out multiple modes in the data, leading to less diverse and lower-quality generations.
### :eyes: Training VAE:

```
cd ./diffusers
python train_vae.py --save-dir="/outputs" --log-dir="/logs" --early-stop-patience=10 --visualize

```
*Note: The script is configured for training based on the CIFAR-10 dataset.*