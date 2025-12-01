# 🧠 Key Assumptions in Denoising Diffusion Probabilistic Models (DDPM)

> Source: *Ho, Jonathan, et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.*

---

## 📋 Table of Contents
1. [Core Mathematical Assumptions](#core-mathematical-assumptions)
2. [Training & Parameterization Assumptions](#training--parameterization-assumptions)
3. [Data & Implementation Assumptions](#data--implementation-assumptions)
4. [Implicit & Derived Assumptions](#implicit--derived-assumptions)
5. [Summary Table](#summary-table)
6. [Assumptions Relaxed in Later Works](#assumptions-relaxed-in-later-works)

---

## Core Mathematical Assumptions

### 1️⃣ Markov Assumption for the Forward Diffusion Process

**Statement:**
- Each diffusion step depends only on the previous one:
  $$
  q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)
  $$
- The full forward process is:
  $$
  q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
  $$

**Key Properties:**
- Gaussian noise is **independent and isotropic** at each step
- This Markov chain gradually transforms real data $x_0$ into pure Gaussian noise $x_T \sim \mathcal{N}(0,I)$
- Enables closed-form sampling: $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I)$

> ✅ **Why this matters:** This assumption allows the process to have a closed-form Gaussian distribution, making it tractable and easy to sample from at any timestep without iterating through all previous steps.

---

### 2️⃣ Gaussian Prior at the Final Step

**Statement:**
- When $T$ is large enough, the distribution $q(x_T \mid x_0)$ becomes nearly standard normal:
  $$
  q(x_T \mid x_0) \approx \mathcal{N}(0, I)
  $$
- Thus, the **prior for the reverse process** is set as:
  $$
  p(x_T) = \mathcal{N}(0, I)
  $$

**Mathematical Justification:**
- When $T \to \infty$ and $\beta_t$ is small, $\bar{\alpha}_T \to 0$
- Therefore: $q(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_T}\,x_0, (1-\bar{\alpha}_T)I) \to \mathcal{N}(0, I)$

**Impact on ELBO:**
- The term $D_{KL}(q(x_T \mid x_0) \| p(x_T))$ in the ELBO **approximates to 0** (not dropped, but negligible)
- This term contains **no learnable parameters**, so it's ignored during optimization

> ✅ **Why this matters:** Starting reverse diffusion from standard Gaussian noise is simple and theoretically justified, eliminating the need to learn the initial distribution.

---

### 3️⃣ Gaussian Markov Assumption for the Reverse Process

**Statement:**
- The reverse process is also modeled as a Markov chain with Gaussian transitions:
  $$
  p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
  $$
- Each reverse step is an isotropic Gaussian:
  $$
  p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
  $$

**Theoretical Foundation:**
- Based on the work of Feller (1949) and subsequent research showing that:
  - If forward steps use small $\beta_t$, the reverse conditional $q(x_{t-1}|x_t)$ is approximately Gaussian
  - The mean of this reverse Gaussian can be learned by a neural network

> ✅ **Why this matters:** This assumption converts the complex task of learning a reverse diffusion process into a simpler Gaussian regression problem, making training feasible with standard deep learning techniques.

---

### 4️⃣ Fixed Variance Assumption

**Statement:**
- Both $q(x_{t-1} \mid x_t, x_0)$ (true posterior) and $p_\theta(x_{t-1} \mid x_t)$ (learned model) share the same variance:
  $$
  \Sigma_\theta(x_t, t) = \sigma_t^2 I = \sigma_q^2(t) I
  $$

**Two Choices for Fixed Variance:**

DDPM experiments with two fixed variance schedules:

1. **Forward process variance:**
   $$
   \sigma_t^2 = \beta_t
   $$

2. **Posterior variance (optimal for $x_0$):**
   $$
   \sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
   $$

Both choices yield similar sample quality empirically.

**Impact on Optimization:**
- KL divergence simplifies to depend only on the means:
  $$
  D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) = \frac{1}{2\sigma_q^2(t)} \| \mu_q(x_t, x_0) - \mu_\theta(x_t, t) \|^2 + \text{const}
  $$

> ✅ **Why this matters:** Fixing variance simplifies the ELBO to a simple MSE loss between means, dramatically improving training stability. The model only needs to learn to predict the mean, not the full covariance structure.

**Note:** Later work (Improved DDPM, 2021) showed that learning variance can improve log-likelihood, but the original fixed-variance approach already produces high-quality samples.

---

### 5️⃣ Isotropic Noise Assumption

**Statement:**
- The covariance matrix at each step is a **scalar multiple of the identity matrix**:
  $$
  \text{Cov}[x_t \mid x_{t-1}] = \beta_t I
  $$
  $$
  \text{Cov}[x_{t-1} \mid x_t] = \sigma_t^2 I
  $$

**Implications:**
- All dimensions are corrupted with the same amount of noise
- No directional preference or correlation structure in the noise
- Noise is added/removed uniformly across all feature dimensions

**Computational Benefits:**
- Reduces parameters from $O(d^2)$ to $O(1)$ per timestep
- Enables efficient sampling using standard Gaussian random vectors
- Simplifies implementation significantly

> ✅ **Why this matters:** While this assumption may seem restrictive, it works remarkably well in practice and avoids the computational burden of learning full covariance matrices, which would be infeasible for high-dimensional data like images.

---

## Training & Parameterization Assumptions

### 6️⃣ Predefined Noise Schedule

**Statement:**
- The sequence of noise coefficients $\{\beta_t\}_{t=1}^T$ is **predefined** (not learned during training)
- Typically satisfies:
  $$
  0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1
  $$

**Common Schedules:**

1. **Linear Schedule (DDPM original):**
   $$
   \beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
   $$
   - Example: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$

2. **Cosine Schedule (Improved DDPM, 2021):**
   $$
   \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
   $$
   - Better preserves information at early timesteps

**Derived Quantities:**
$$
\alpha_t = 1-\beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

> ✅ **Why this matters:** A carefully designed noise schedule ensures smooth and stable diffusion. It guarantees that by $t=T$, the data distribution has been transformed into approximately standard Gaussian noise, enabling the reverse process to start from a simple distribution.

---

### 7️⃣ Mean Parameterization via Noise Prediction

**Statement:**
- Instead of directly predicting the mean $\mu_\theta(x_t, t)$, DDPM parameterizes it through **noise prediction**

**Three Equivalent Parameterizations:**

The reverse mean can be parameterized in three mathematically equivalent ways:

1. **Direct mean prediction:**
   $$
   p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
   $$

2. **Noise prediction (DDPM's choice):**
   $$
   \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t,t)\right)
   $$
   The network predicts the noise $\varepsilon$ that was added

3. **Original data prediction:**
   $$
   \mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_\theta(x_t,t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
   $$
   The network predicts the clean data $x_0$

**Why Noise Prediction?**

DDPM chooses noise prediction because:
- **Empirically superior:** Experiments show it converges faster and produces better samples
- **Stable gradients:** Predicting noise (zero-mean, unit variance) is easier than predicting data (arbitrary distribution)
- **Connection to score matching:** $\varepsilon_\theta \approx -\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log q(x_t)$

**Resulting Loss:**
$$
L_{\text{simple}} = \mathbb{E}_{x_0, \varepsilon \sim \mathcal{N}(0,I), t}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]
$$

where $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$

> ✅ **Why this matters:** This converts the complex task of generative modeling into a simple denoising task—predicting and removing noise. This is conceptually intuitive and empirically very effective.

---

### 8️⃣ Shared Network Across All Timesteps

**Statement:**
- A **single neural network** $\varepsilon_\theta(x_t, t)$ is used for all timesteps $t \in \{1, 2, ..., T\}$
- The timestep $t$ is provided as an additional input, typically through embedding

**Implementation Details:**
- Timestep embedding: $t \to [\sin(t\omega_1), \cos(t\omega_1), \sin(t\omega_2), \cos(t\omega_2), ...]$
- Architecture: Typically U-Net with:
  - Timestep conditioning at each resolution level
  - Self-attention layers for capturing long-range dependencies
  - Residual connections

**Benefits:**
- **Parameter efficiency:** One model instead of $T$ separate models
- **Generalization:** Network learns patterns that apply across different noise levels
- **Flexibility:** Can evaluate at arbitrary timesteps (useful for DDIM and other fast samplers)

> ✅ **Why this matters:** This architectural choice enables the model to learn general denoising patterns rather than memorizing specific noise levels, leading to better generalization and more efficient use of model capacity.

---

### 9️⃣ Simplified Training Objective

**Statement:**
- DDPM does **not** optimize the full variational lower bound (ELBO) with proper weighting

**Full ELBO:**
$$
L_{\text{vlb}} = L_0 + L_1 + L_2 + ... + L_{T-1} + L_T
$$

where:
$$
L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))
$$

Each $L_t$ should theoretically be weighted by $\frac{1}{2\sigma_t^2}$.

**Simplified Objective Used in Practice:**
$$
L_{\text{simple}} = \mathbb{E}_{t \sim \text{Uniform}\{1,...,T\}}\left[L_t\right]
$$

**Key Simplifications:**
1. **Uniform timestep sampling:** Each $t$ is equally likely during training
2. **Remove weighting:** Drop the $\frac{1}{2\sigma_t^2}$ coefficient
3. **Single timestep per batch:** Sample one $t$ per training example

**Impact:**
- The simplified objective **doesn't directly optimize log-likelihood**
- But empirically produces **better sample quality** than the weighted ELBO
- Trade-off: slightly worse log-likelihood metrics, much better perceptual quality

**Intuition:**
- Uniform weighting gives equal importance to all noise levels
- Weighted ELBO overemphasizes early timesteps (small noise)
- Simplified version forces the model to handle all noise levels equally well

> ✅ **Why this matters:** This is one of the most important practical insights in DDPM—ignoring the theoretically "correct" weighting actually improves results! It shows that optimizing for log-likelihood and optimizing for sample quality are not always aligned.

---

## Data & Implementation Assumptions

### 🔟 Data Normalization and Space

**Statement:**
- Input data $x_0$ is **normalized** to a fixed range, typically:
  - $x_0 \in [-1, 1]^d$ (most common), or
  - $x_0 \in [0, 1]^d$
- The diffusion process operates in **continuous Euclidean space** $\mathbb{R}^d$

**Normalization Procedure:**
For images with pixel values in $[0, 255]$:
$$
x_0 = \frac{\text{image}}{127.5} - 1 \quad \text{(to get range [-1,1])}
$$

**Why Normalization Matters:**
- Ensures compatibility with Gaussian noise assumptions
- Stabilizes training gradients
- Makes $\beta_t$ values meaningful across different datasets
- Final samples can be denormalized: $\text{image} = 127.5 \times (x_0 + 1)$

**Continuous Space Assumption:**
- DDPM treats images as continuous-valued (not discrete 0-255 integers)
- This enables smooth gradient flow during training
- For discrete data (text, categorical), modifications are needed (e.g., D3PM)

> ✅ **Why this matters:** Proper data preprocessing is crucial for diffusion models. The Gaussian noise assumption only makes sense when data is appropriately normalized to a bounded range.

---

### 1️⃣1️⃣ Number of Diffusion Steps $T$

**Statement:**
- The number of steps $T$ must be **large enough** so that:
  $$
  q(x_T|x_0) \approx \mathcal{N}(0, I) \quad \text{for all } x_0
  $$
- Equivalently: $\bar{\alpha}_T \approx 0$

**Typical Values:**
- DDPM original: $T = 1000$
- Must be large relative to $\beta_t$ values
- With $\beta_1 = 10^{-4}$, $\beta_T = 0.02$ (linear), $\bar{\alpha}_{1000} \approx 0.0001$

**Training vs Inference:**
- **Training:** Always uses all $T$ steps
- **Inference:** Can use fewer steps via:
  - Strided sampling (every $k$-th step)
  - DDIM (deterministic, non-Markovian)
  - DPM-Solver (numerical ODE solver)
  - Typically 20-50 steps for good quality

**Trade-offs:**
- Larger $T$: Better approximation, slower sampling
- Smaller $T$: Faster but may not fully destroy structure
- Modern methods (DDIM, DPM) allow $T_{\text{train}} = 1000$, $T_{\text{sample}} = 50$

> ✅ **Why this matters:** The choice of $T$ directly affects both the quality of the Gaussian approximation and the computational cost. Too small and the reverse process may not converge to the data distribution; too large and sampling becomes impractically slow.

---

### 1️⃣2️⃣ Tractable Data Distribution Assumption

**Statement:**
- The true data distribution $p_{\text{data}}(x_0)$ is assumed to have **sufficient structure** that can be learned by:
  - A neural network with finite capacity
  - Through the diffusion modeling framework

**Implicit Assumptions:**
- Data lies on or near a lower-dimensional manifold in $\mathbb{R}^d$
- The manifold structure can be captured by iterative denoising
- There exists a function class (e.g., U-Nets) that can approximate the score function

**Practical Considerations:**
- Works well for natural images (strong local correlations)
- Works well for audio waveforms (temporal structure)
- May struggle with:
  - Purely random data (no structure)
  - Very high-frequency details
  - Sparse, discrete structures

> ✅ **Why this matters:** Diffusion models work because natural data has structure. If the data were truly random noise, the model would have nothing to learn. This assumption, while rarely stated explicitly, is fundamental to why diffusion models succeed on real-world data.

---

## Implicit & Derived Assumptions

### 1️⃣3️⃣ Sampling Procedure

**Statement:**
- During inference, samples are generated by **iteratively applying the reverse process**:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t,t)\right) + \sigma_t z_t
$$

where:
- $z_t \sim \mathcal{N}(0, I)$ for $t > 1$ (stochastic)
- $z_t = 0$ for $t = 1$ (deterministic final step)

**Algorithm:**
```
1. Sample x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
3.     z = N(0, I) if t > 1 else 0
4.     x_{t-1} = mean + sigma * z
5. return x_0
```

**Key Properties:**
- **Ancestral sampling:** Each step depends on previous step (Markovian)
- **Stochastic:** Randomness from $z_t$ adds diversity
- **Gradual refinement:** Goes from noise to structure over T steps

**Variants:**
- **DDIM:** Sets $\sigma_t = 0$ (deterministic)
- **DDPM:** Uses $\sigma_t = \sqrt{\beta_t}$ or $\sqrt{\tilde{\beta}_t}$
- Can control stochasticity via $\eta$ parameter: $\sigma_t = \eta\sqrt{\tilde{\beta}_t}$

> ✅ **Why this matters:** The sampling procedure is where the learned model is actually used to generate data. Understanding the role of stochasticity vs determinism is key to controlling sample quality and diversity.

---

### 1️⃣4️⃣ Model Capacity and Convergence

**Statement:**
- The neural network $\varepsilon_\theta$ has **sufficient capacity** to approximate the true noise at each timestep:
  $$
  \varepsilon_\theta(x_t, t) \approx \varepsilon_{\text{true}}
  $$

**Practical Requirements:**
- Large enough architecture (e.g., U-Net with 100M+ parameters for ImageNet)
- Appropriate receptive field (must see relevant context)
- Enough training data and iterations

**Convergence Assumption:**
- With sufficient training, the loss $L_{\text{simple}}$ converges
- The learned model approximates the true reverse process
- No guarantee of global optimum (deep learning is non-convex)

> ✅ **Why this matters:** Like all deep learning, DDPM relies on the universal approximation capabilities of neural networks. In practice, architecture choice and training procedure are critical for success.

---

### 1️⃣5️⃣ Independence of Noise Samples

**Statement:**
- At each timestep during training:
  $$
  \varepsilon \sim \mathcal{N}(0, I) \quad \text{is sampled independently}
  $$
- Noise at different timesteps is uncorrelated
- Noise is independent of the data $x_0$

**Implications:**
- Enables parallel training across timesteps
- Simplifies the theoretical analysis
- Each training step is an independent sample from the loss distribution

> ✅ **Why this matters:** This independence assumption allows efficient training where each mini-batch can randomly sample both data and timesteps without worrying about correlations.

---

## 📘 Summary Table

| # | Assumption | Mathematical Form | Purpose | Can be Relaxed? |
|---|------------|-------------------|---------|-----------------|
| 1 | Markov forward process | $q(x_t\|x_{t-1}) = \mathcal{N}(...)$ | Tractable closed-form | ✅ Yes (SDE) |
| 2 | Gaussian prior | $p(x_T) = \mathcal{N}(0,I)$ | Simple initialization | Rarely needed |
| 3 | Gaussian reverse | $p_\theta(x_{t-1}\|x_t) = \mathcal{N}(...)$ | Simple generative model | ✅ Yes (discrete) |
| 4 | Fixed variance | $\Sigma_\theta = \sigma_q^2 I$ | Simplifies to MSE | ✅ Yes (I-DDPM) |
| 5 | Isotropic noise | $\text{Cov} = \sigma^2 I$ | Reduces parameters | Difficult |
| 6 | Fixed noise schedule | $\beta_t$ predefined | Smooth progression | ✅ Yes (learnable) |
| 7 | Noise prediction | $\varepsilon_\theta(x_t,t)$ | Stable training | ✅ Yes (x₀ pred) |
| 8 | Shared network | One $\varepsilon_\theta$ for all $t$ | Parameter efficiency | Possible |
| 9 | Simplified loss | Uniform $t$ weighting | Better samples | Trade-off |
| 10 | Normalized data | $x_0 \in [-1,1]$ | Gradient stability | Required |
| 11 | Large $T$ | $T = 1000$ | Gaussian approx | ✅ Yes (DDIM) |
| 12 | Data structure | Manifold hypothesis | Learnability | Inherent |
| 13 | Ancestral sampling | Iterative $x_T \to x_0$ | Generation method | ✅ Yes (DDIM) |
| 14 | Model capacity | NN can approximate | Convergence | Required |
| 15 | Noise independence | $\varepsilon \perp t, x_0$ | Training efficiency | Required |

---

## 🔄 Assumptions Relaxed in Later Works

### **Improved DDPM (Nichol & Dhariwal, ICML 2021)**

**Relaxed Assumptions:**
- ❌ Fixed variance → ✅ Learned variance
  $$
  \Sigma_\theta(x_t,t) = \exp(v_\theta(x_t,t)) \cdot I
  $$
- ❌ Linear noise schedule → ✅ Cosine schedule
- ❌ Simplified loss → ✅ Hybrid loss (simple + vlb)

**Results:**
- Improved log-likelihood (better density modeling)
- Comparable or slightly better sample quality
- More theoretically principled

---

### **DDIM (Song et al., ICLR 2021)**

**Relaxed Assumptions:**
- ❌ Markov reverse process → ✅ Non-Markovian (deterministic)
- ❌ Stochastic sampling → ✅ Deterministic with $\sigma_t = 0$
- ❌ T-step sampling required → ✅ Fast sampling (20-50 steps)

**Key Innovation:**
$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted }x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}}\varepsilon_\theta(x_t,t)}_{\text{direction pointing to }x_t}
$$

**Results:**
- 10-50× faster sampling
- Deterministic generation (same $x_T$ → same $x_0$)
- Enables interpolation in latent space

---

### **Latent Diffusion Models / Stable Diffusion (Rombach et al., CVPR 2022)**

**Relaxed Assumptions:**
- ❌ Diffusion in pixel space → ✅ Diffusion in latent space
- Data assumption: $x_0 \in \mathbb{R}^{H \times W \times 3}$ → $z_0 \in \mathbb{R}^{h \times w \times c}$

**Architecture:**
```
x_0 → [VAE Encoder] → z_0 → [Diffusion Process] → z_T
z_T → [Reverse Diffusion] → ẑ_0 → [VAE Decoder] → x̂_0
```

**Benefits:**
- 4-8× reduction in memory and compute
- Enables high-resolution generation (1024×1024+)
- Faster training and sampling
- Easy to add conditioning (text, class, etc.)

---

### **Score-Based Models / SDE (Song et al., ICLR 2021)**

**Relaxed Assumptions:**
- ❌ Discrete timesteps → ✅ Continuous time $t \in [0,1]$
- ❌ DDPM formulation → ✅ Stochastic Differential Equation (SDE)

**Forward SDE:**
$$
dx = f(x,t)dt + g(t)dw
$$

**Reverse SDE:**
$$
dx = [f(x,t) - g(t)^2\nabla_x\log p_t(x)]dt + g(t)d\bar{w}
$$

**Key Insight:**
- DDPM is a discretization of the Variance Preserving (VP) SDE
- Enables flexible sampling via ODE/SDE solvers
- Unifies score matching and diffusion models

**Results:**
- More flexible noise schedules
- Better theoretical understanding
- Advanced sampling methods (Probability Flow ODE)

---

### **EDM (Karras et al., NeurIPS 2022)**

**Relaxed/Clarified Assumptions:**
- ❌ Specific noise schedule → ✅ Optimal preconditioning
- Analyzes design space systematically:
  - Noise schedule: $\sigma(t)$ parametrization
  - Preconditioning: input/output scaling
  - Loss weighting: $\lambda(t)$ function
  - Sampling: optimal ODE solver

**Key Contributions:**
- Proposes "best practices" for diffusion training
- Shows many design choices are suboptimal
- Achieves SOTA with proper design

---

### **Consistency Models (Song et al., ICML 2023)**

**Completely Different Paradigm:**
- ❌ Iterative sampling → ✅ Single-step generation
- ❌ Learns noise prediction → ✅ Learns consistency function
- Can still do multi-step for quality

**Key Property:**
$$
f_\theta(x_t, t) = f_\theta(x_{t'}, t') = x_0 \quad \forall t, t'
$$

**Results:**
- 1-step generation competitive with 10-step DDIM
- Can trade steps for quality
- Faster than any diffusion model

---

### **Rectified Flow (Liu et al., ICLR 2023)**

**Relaxed Assumptions:**
- ❌ Curved diffusion paths → ✅ Straight interpolation paths
- ❌ Gaussian noise → ✅ Any coupling

**Key Idea:**
Learn the **straightest possible path** from noise to data:
$$
\frac{dx_t}{dt} = v_\theta(x_t, t), \quad x_0 \sim p_{\text{data}}, \quad x_1 \sim p_{\text{noise}}
$$

**Benefits:**
- Faster sampling (fewer steps needed)
- Better theoretical properties
- Easier to train

---

## 🎯 Key Takeaways

### What Makes DDPM Work?

1. **Smart decomposition:** Breaking generation into T small denoising steps
2. **Gaussian assumption:** Makes each step tractable and learnable
3. **Simple objective:** MSE loss is stable and effective
4. **Architectural choice:** U-Net with attention captures hierarchical structure
5. **Empirical tuning:** Many "suboptimal" choices work better in practice

### What Can Be Changed?

- ✅ **Safe to change:** Variance schedule, parameterization, number of training steps
- ⚠️ **Carefully:** Variance learning, loss weighting, architecture
- ❌ **Hard to change:** Gaussian assumption, isotropy (without major redesign)

### Philosophy

DDPM shows that:
- **Simplicity often wins** over theoretical optimality
- **Iterative refinement** is a powerful paradigm
- **Strong inductive biases** (Gaussian, Markov) enable learning
- **Empirical validation** matters more than perfect theory

---

## 📚 References

### Original Papers

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
   
2. **Sohl-Dickstein, J., et al. (2015).** "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML 2015*.

### Key Improvements

3. **Nichol, A. Q., & Dhariwal, P. (2021).** "Improved Denoising Diffusion Probabilistic Models." *ICML 2021*.

4. **Song, J., Meng, C., & Ermon, S. (2020).** "Denoising Diffusion Implicit Models." *ICLR 2021*.

5. **Rombach, R., et al. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.

6. **Song, Y., et al. (2021).** "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.

7. **Karras, T., et al. (2022).** "Elucidating the Design Space of Diffusion-Based Generative Models." *NeurIPS 2022*.

8. **Song, Y., et al. (2023).** "Consistency Models." *ICML 2023*.

9. **Liu, X., et al. (2023).** "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*.

### Theoretical Foundations

10. **Feller, W. (1949).** "On the Theory of Stochastic Processes, with Particular Reference to Applications." *Proceedings of the Berkeley Symposium on Mathematical Statistics and Probability*.

11. **Anderson, B. D. (1982).** "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*.

### Additional Resources

- **Lilian Weng's Blog:** "What are Diffusion Models?" - Excellent tutorial
- **Yang Song's Blog:** "Generative Modeling by Estimating Gradients of the Data Distribution"
- **Hugging Face Diffusers Documentation:** Practical implementation guide
- **Annotated Diffusion Model:** https://huggingface.co/blog/annotated-diffusion

---

*Last updated: November 2024*  
*Based on DDPM (Ho et al., NeurIPS 2020) and subsequent improvements (2020-2023)*