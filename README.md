# GSoc-2026-ML4SCI-task2b
**Task 2b** evaluation test for Google Summer of Code 2026, under the [ML4SCI](https://ml4sci.org/) umbrella organization, [E2E / CMS](https://ml4sci.org/gsoc/2026/proposal_E2E1.html) project.

**Author:** Yuxi Yang — Johns Hopkins University

## Task

Train a GAN-based super-resolution model to upsample low-resolution (64×64) calorimeter jet images to high-resolution (125×125) for the CMS detector at the LHC.

## Approach

**Model:** Enhanced Super-Resolution GAN（ESRGAN） with RRDB blocks

**Architecture:**
- Generator: Conv → 6× RRDB → PixelShuffle 2× (64→128) → Bilinear resize (128→125) → Conv output
- Discriminator: VGG-style with relativistic average loss
- No Batch Normalization in the generator

**Loss Function:**
- L1 pixel loss (λ=1.0) — dominant, to preserve physical accuracy of energy deposits
- VGG perceptual loss (λ=0.006) — structural detail recovery
- Adversarial loss (λ=0.001) — high-frequency sharpness

**Training:** L1 pre-training (10 epochs) followed by GAN training (40 epochs)

## Results

| Metric | ESRGAN (Ours) |
|--------|--------------|
| PSNR   | 76.02 ± 0.08 dB |
| SSIM   | 0.9998 ± 0.0000 |

### Training Curves
<img width="2383" height="581" alt="training_curves" src="https://github.com/user-attachments/assets/13ec82ae-a2d9-4309-a750-d15d6a2cd1b5" />


### LR vs HR Samples
<img width="2291" height="889" alt="lr_hr_comparison" src="https://github.com/user-attachments/assets/99b51cee-6f34-4b46-9b8c-954e9dc25bce" />


### Super Resolution Comparison (LR → Bicubic → SR → HR)
<img width="2645" height="1771" alt="sr_visual_comparison" src="https://github.com/user-attachments/assets/6a8b7171-93a4-4795-8ddd-d045c121051b" />


### Energy Distribution (SR vs HR)
<img width="2235" height="581" alt="energy_distribution" src="https://github.com/user-attachments/assets/b511d625-cdab-446f-a192-e0a5bea430f6" />


## Dataset

- **Source:** CMS simulated QCD jet images (quarks and gluons)
- **LR:** 3 × 64 × 64 | **HR:** 3 × 125 × 125
- **Samples used:** 10,000 (train 8,000 / val 1,000 / test 1,000)
- **Format:** Parquet files with columns `X_jets_LR`, `X_jets`, `pt`, `m0`, `y`

## Discussion

The high PSNR/SSIM values reflect the extreme sparsity of calorimeter data that most pixels are near zero, and the model reconstructs these accurately. As confirmed by the project mentors, **physical accuracy of particle positions and energies/momenta is the primary evaluation criterion**. Standard image metrics like SSIM are less suited for this domain due to their scale insensitivity, which could mask bias shifts in background noise.

This insight motivates several directions for improvement:
- **Physics-specific metrics**: per-jet energy conservation ratio, spatial centroid accuracy, and energy spectrum fidelity would better capture reconstruction quality than PSNR/SSIM alone.
- **Physics-informed losses**: incorporating energy conservation constraints or detector response functions directly into the training objective.
- **Modern architectures**: as specified in the project description, Visual Autoregression, JEPA, and diffusion-based SR should be explored for the full GSoC project.
- **Graph-based approaches**: following [arXiv:2409.16052](https://arxiv.org/abs/2409.16052), treating calorimeter hits as graphs rather than images could better handle the sparse, irregular nature of the data.

## Requirements

```
torch, torchvision, numpy, pyarrow, scikit-image, matplotlib
```

## References

- [Denoising Graph SR for Collider Reconstruction (arXiv:2409.16052)](https://arxiv.org/abs/2409.16052)
- [SuperCalo: Calorimeter Shower Super-Resolution (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
- [DiffLense: Conditional Diffusion SR (NeurIPS ML4PS 2024)](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_89.pdf)
