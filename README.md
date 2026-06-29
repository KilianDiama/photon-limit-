# Certified Reconstruction Engine – Photon-Limited Imaging (PyTorch)

A unified and certified reconstruction pipeline for photon-limited imaging, combining:

- A stabilized Poisson–Gaussian forward model  
- Anisotropic reaction–diffusion reconstruction (MRA)  
- Functional L‑BFGS optimization (OCA)  
- Cramér–Rao uncertainty quantification (QIA)  

The entire system is mutation‑free, fully differentiable, `torch.compile`‑friendly, and designed for industrial‑grade scientific imaging.

---

## 🔍 Overview

Photon-limited imaging requires robust forward models, stable optimization, and reliable uncertainty quantification.  
This engine provides a complete reconstruction workflow:

1. **ForwardModel** — Poisson–Gaussian likelihood with Softplus stabilization  
2. **Pure geometric operators** — gradient, divergence, and TV proximal without in‑place ops  
3. **MRAReconstructor** — anisotropic reaction–diffusion solver  
4. **OCAOptimizer** — vectorized L‑BFGS with static memory  
5. **UncertaintyQuantifier** — Cramér–Rao Bound (CRB) via adjoint Fisher  
6. **CertifiedReconstructionEngine** — unified pipeline combining all components

---

## 🧩 Components

### 1. ForwardModel
Implements the direct operator, adjoint operator, simulation of photon-limited measurements, and the negative log-likelihood.

**Features**
- Poisson–Gaussian noise model  
- Softplus-stabilized likelihood  
- Safe clamping to avoid numerical instabilities  
- Synthetic data generation (`simulate`)  

---

### 2. Pure Geometric Operators

All operators are compiled with `torch.compile(fullgraph=True)` and contain **zero in‑place mutation**.

- `grad_image_pure(u)` — functional 2D gradient  
- `div_image_pure(g)` — exact adjoint divergence  
- `prox_tv_pure(u, lam)` — Chambolle’s TV projection (fully functional)

These operators form the geometric backbone of the reconstruction pipeline.

---

### 3. MRAReconstructor — Reaction–Diffusion

A hybrid solver combining:
- Gradient descent on the data fidelity  
- Anisotropic diffusion guided by gradient magnitude  
- TV proximal regularization  

This module provides a stable and physically meaningful reconstruction baseline.

---

### 4. OCAOptimizer — Functional L‑BFGS

A fully vectorized L‑BFGS optimizer:
- No dynamic loops  
- No in‑place mutation  
- Static memory rank  
- Compatible with `torch.compile`  

It refines the MRA output using second-order information.

---

### 5. UncertaintyQuantifier — Cramér–Rao Bound

Computes a global CRB estimate using:
- Fisher information in measurement space  
- Projection via adjoint operator  
- Stabilized trace approximation  

Provides a certified uncertainty bound on the reconstruction.

---

### 6. CertifiedReconstructionEngine

The main orchestrator:
1. Runs MRA reconstruction  
2. Runs OCA optimization  
3. Computes CRB uncertainty  

**Output dictionary**
```python
{
    "reconstruction": final_image,
    "mra_reconstruction": intermediate_image,
    "error_bound": crb_value
}
