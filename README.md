# JailDAM Autoencoder For Jailbreak Detection

This repository contains the three autoencoder-based baselines for jailbreak detection.

## Autoencoder Baselines

### Quick Start
```bash
# Run all three approaches
python autoencoder_vlm_baseline.py      # Baseline
python autoencoder_dual.py              # Dual autoencoder approach

# Compare results
python compare_approaches.py
```

### Approaches
1. **Unsupervised**: Trains only on benign data, detects anomalies via reconstruction error
3. **Dual**: Trains separate autoencoders for benign and unsafe data, uses error difference

## 📁 Repository Structure

### Autoencoder Implementation
- `Autoencoder.py` - Enhanced autoencoder class
- `autoencoder_vlm_baseline.py` - Unsupervised baseline
- `autoencoder_dual.py` - Dual autoencoder approach

### Dataset Configuration
- `reference/load_datasets.py` - Dataset loading
- `data` - Dataset files (organized by source, not shown in this repository)

### Results
- `results/` - Experiment results (JSON format)

### Detection Method Details

#### 1. Unsupervised Autoencoder
- **Training**: Autoencoder learns to reconstruct benign patterns only
- **Assumption**: Benign data has consistent patterns; unsafe data will have high reconstruction error
- **Detection**: `MSE_Loss(input, reconstructed) > threshold → Unsafe`
- **Threshold**: Optimized using validation subset for best F1 score

#### 2. Dual Autoencoder
- **Training**:
  - Benign autoencoder: Trained exclusively on benign data
  - Unsafe autoencoder: Trained exclusively on unsafe data
  - Each model specializes in reconstructing its respective data type
- **Detection**:
  - Benign samples: Low benign error, high unsafe error → Negative score
  - Unsafe samples: High benign error, low unsafe error → Positive score
  - Detection score: `Benign_Error - Unsafe_Error`
  - Positive score → Unsafe, Negative score → Benign

Results are saved to `results/` directory with detailed per-dataset breakdowns.

## 📜 Citing JailDAM
Part of the implementation refers to their [original repository](https://github.com/ShenzheZhu/JailDAM). If by any chance this repository is useful to you, you only need to cite their work:

```bibtex
@article{nian2025jaildam,
  title={JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model},
  author={Nian, Yi and Zhu, Shenzhe and Qin, Yuehan and Li, Li and Wang, Ziyi and Xiao, Chaowei and Zhao, Yue},
  journal={arXiv preprint arXiv:2504.03770},
  year={2025}
}
```
