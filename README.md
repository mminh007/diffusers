# 💪 Stable Diffusion: 

 *This repository focuses on analyzing each component of Stable Diffusion.*

```
git pull https://github.com/mminh007/diffusers.git
```
---

## 🚀 Variant-AutoEncoder:

🔗 <a href= "/VAE/README.md"> Explained of VAE  </a>

Variational autoencoders (VAE) are deep learning models composed of an encoder that learns to isolate the important latent variables from training data and a decoder that then uses those latent variables to reconstruct the input data.

However, whereas most autoencoder architectures encode a discrete, fixed representation of latent variables, VAEs encode a continuous, probabilistic representation of that latent space. This enables a VAE to not only accurately reconstruct the exact original input, but also use variational inference to generate new data samples that resemble the original input data.

VAEs are a subset of the larger category of autoencoders, a neural network architecture typically used in deep learning for tasks such as data compression, image denoising, anomaly detection and facial recognition.

💪 **Strongness:**

✅ **Probabilistic Framework** – Unlike traditional autoencoders, VAE learns a probabilistic distribution rather than a fixed mapping, improving robustness and diversity in generated samples.

✅ **Continuous and Structured Latent Space** – VAE enforces a smooth and continuous latent space, making it well-suited for generating coherent and meaningful samples.

✅ **Scalability** – VAEs are relatively efficient and scalable to high-dimensional data, such as images and text.

✅ **Uncertainty Estimation** – Since VAEs are probabilistic models, they provide uncertainty estimates, which are useful in applications like anomaly detection.

👎 **Weakeness:** 

✅ One of the biggest weaknesses is **the low reconstruction quality, as generated images often appear blurry**. This arises from the trade-off between the two main components in the loss function: **Reconstruction Loss** and **KL Divergence**. These two components often oppose each other, leading to a trade-off that results in reconstructions that are not as sharp as those produced by other models like GANs.

✅ **Limited Expressiveness of Latent Space** – The assumption that the latent variables follow a Gaussian distribution can sometimes restrict the model’s ability to capture complex structures in the data.

✅ **Mode Averaging** – Unlike GANs, which can model sharper distributions, VAEs tend to average out multiple modes in the data, leading to less diverse and lower-quality generations.

### 👀 Training VAE:

```
cd ./diffusers
python train_vae.py --save-dir="/outputs" --log-dir="/logs" --early-stop-patience=10 --visualize

```
*Note: The script is configured for training based on the CIFAR-10 dataset.*

---

## 🚀 Denoising Diffusion Probabilistic Model (DDPM)

🔗 <a href= "#"> Explained of Diffusion Process  </a>

🔗 <a href="/diffusion/Assumptions-DDPM.md"> Assumpition applied in DDPM </a>

In deep learning, **"diffusion"** refers to a class of generative models that learn to generate data by progressively adding noise to a dataset and then learning to reverse this process, effectively removing the noise and reconstructing the data or creating new, realistic versions.

**Diffusion models** are a type of generative model, meaning they are trained to create new data samples that resemble the data they were trained on. Diffusion models operate in two main steps:

✅ **Forward Process (Noise Addition):** The process starts with real data and progressively adds noise to it, gradually transforming the data into pure noise. 
-   The model gradually adds Gaussian noise to the input data (e.g., an image) over multiple steps.
-   After many steps, the data becomes pure noise.
-   This process is mathematically modeled as a **Markov Chain** using a stochastic differential equation.

✅ **Reverse Process (Denoising):** The model learns to reverse this process by training a neural network to convert noise back into data, effectively removing the noise step-by-step, reconstructing the original data from noise. 
-   The model **learns to reverse the noise step-by-step** to reconstruct the original data.
-   A neural network (usually a **U-Net architecture**) predicts how to remove the noise at each step.
-   This reverse process generates high-quality images from pure noise.

---

### Tasks and Pipelines

|**Task**|**Pipeline**|
|--------|------------|
|Text-to-image-lora|<a href= "./examples/text_to_image/README.md"> Link</a>|

---

## 👀 Papers 

| Model / Paper  | Gaussian Assumption | Description &nbsp; | Link |
|---------------|----------------------|-------------|-------|
| **VAE (Kingma & Welling, 2013)** | ✔ Gaussian latent | Assumes prior $p(z)=\mathcal{N}(0,I)$.<br> Posterior $q_\phi(z\|x)$ Gaussian. Optimizes ELBO to approximate data distribution. | https://arxiv.org/abs/1312.6114 |
| **NCSN – Denoising Score Matching (2019)** | ✔ Multiple Gaussian noise levels | Uses noise scales $\sigma_1 > \dots > \sigma_L$.<br> Learns score $\nabla_x \log p_\sigma(x)$. Sampling via Annealed Langevin Dynamics. | https://arxiv.org/abs/1907.05600 |
| **DDPM (Ho et al., 2020)** | ✔ Gaussian preserved | Forward: $q(x_t\|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_tI)$.<br> Reverse Gaussian predicted by neural network. | https://arxiv.org/abs/2006.11239 |
| **Improved DDPM (2021)** | ✔ Gaussian (learned variance) | Same Gaussian forward; reverse uses learned variance and cosine $\beta_t$ schedule → better FID. | https://arxiv.org/abs/2102.09672 |
| **DDIM (Song et al., 2020)** | ✔ Gaussian in training; ✖ no Gaussian sampling | Same training as DDPM; sampling uses non-Markovian deterministic ODE-like reverse process. | https://arxiv.org/abs/2010.02502 |
| **Score-based SDE (2020–2021)** | ✖ Not discrete Gaussian chain | Continuous-time SDEs (VPSDE, VESDE).<br> Noise from Brownian motion (Gaussian kernel) but schedule no longer discrete. | https://arxiv.org/abs/2011.13456 / https://arxiv.org/abs/2011.09665 |
| **D3PM – Discrete Diffusion (2021)** | ✖ Non-Gaussian | Forward is categorical / multinomial transitions; reverse learned by network. | https://arxiv.org/abs/2107.03006 |
| **Latent Diffusion – LDM (2022)** | ✔ Gaussian in latent space | Images compressed by VAE → diffusion operates on latent $z$ with Gaussian noise → huge compute savings. | https://arxiv.org/abs/2112.10752 |
| **Classifier Guidance (2021)** | ✔ Gaussian unchanged | Adds classifier gradient to adjust reverse dynamics; forward Gaussian intact. | https://arxiv.org/abs/2105.05233 |
| **Classifier-Free Guidance (2022)** | ✔ Gaussian unchanged | Mixes conditional & unconditional predictions to shift score field. | https://arxiv.org/abs/2207.12598 |
| **EDM (Karras et al., 2022)** | ✖ Non-fixed Gaussian noise | Redesigns noise schedule in $\sigma$-space; unifies diffusion, SDE, consistency approaches. | https://arxiv.org/abs/2206.00364 |
| **Flow Matching (2023)** | ✖ No Gaussian forward | Learns vector field $v_t(x)$ to transport noise → data via ODE; no Markov chain. | https://arxiv.org/abs/2209.00796 |
| **Rectified Flow (2023)** | ✖ Non-Gaussian | Defines deterministic straight flow between noise and data; train using vector field regression. | https://arxiv.org/abs/2305.08891 |
| **Consistency Models (2023)** | ✖ No Gaussian Markov chain | Learns a consistency function $f(x_t,t)$ enabling 1–few-step sampling. | https://arxiv.org/abs/2303.01469 |
| **Diffusion Transformer – DiT (2023)** | ✔ Gaussian forward | Uses Vision Transformer block for DDPM-style reverse process; forward Gaussian. | https://arxiv.org/abs/2212.09748 |
| **Stable Diffusion 3 / FLUX (2024)** | ✖ Flow-matching + cross-attention | Large-scale T2I model using flow-matching rather than DDPM-style forward Gaussian. | https://arxiv.org/abs/2403.03206 |
| **Efficient Diffusion Models Survey (2024)** | — Survey | Large survey of diffusion, SDE, flow-matching, consistency, distillation, efficient inference. | https://arxiv.org/abs/2412.05832 |
| **Self-Corrected Flow Distillation (2024)** | ✖ Flow-based | Distills flow models for 1-step or few-step high-quality T2I sampling. | https://arxiv.org/abs/2412.16906 |
| **Diff2Flow (2025)** | ✖ Flow-matching | Converts pretrained diffusion → flow matching for faster inference. | https://arxiv.org/abs/2506.02221 |
| **SenseFlow (2025)** | ✖ Flow-based | Distillation framework for large flow-matching T2I models (e.g., FLUX, SD3.5). | https://arxiv.org/abs/2506.00523 |
| **SANA-Sprint (ICCV 2025)** | ✖ Consistency + diffusion | One-step diffusion via continuous-time consistency distillation. | https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf |
| **Intro to Flow Matching & Diffusion (2025)** | — Tutorial | Mathematical bridge linking diffusion ↔ flow matching ↔ SDE. | https://arxiv.org/abs/2506.02070 |