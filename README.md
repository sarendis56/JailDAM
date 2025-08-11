# Jail-DAM with Autoencoder Baselines

This repository contains the original Jail-DAM implementation plus three autoencoder-based baselines for jailbreak detection.

## Autoencoder Baselines

### Quick Start
```bash
# Run all three approaches
python autoencoder_vlm_baseline.py      # Unsupervised baseline
python autoencoder_semisupervised.py    # Semi-supervised approach
python autoencoder_dual.py              # Dual autoencoder approach

# Compare results
python compare_approaches.py
```

### Approaches
1. **Unsupervised**: Trains only on benign data, detects anomalies via reconstruction error
2. **Semi-Supervised**: Uses both benign and unsafe training data with combined objectives
3. **Dual**: Trains separate autoencoders for benign and unsafe data, uses error difference

## 📁 Repository Structure

### Autoencoder Implementation
- `Autoencoder.py` - Enhanced autoencoder class
- `autoencoder_vlm_baseline.py` - Unsupervised baseline
- `autoencoder_semisupervised.py` - Semi-supervised approach
- `autoencoder_dual.py` - Dual autoencoder approach

### Dataset Configuration
- `reference/load_datasets.py`
- `reference/reference_load_datasets.py` - Original dataset loading code in my approach. *Not runnable standalone.*
- `data` - Dataset files (organized by source, not shown in this repository)

### Results
- `results/` - Experiment results (JSON format)

## 📊 Performance Comparison

The three autoencoder approaches provide different trade-offs:

| Approach | Training Data | Detection Method | Best For |
|----------|---------------|------------------|----------|
| Unsupervised | Benign only | Reconstruction error | Realistic scenarios |
| Semi-Supervised | Benign + Unsafe | Combined score | Maximum performance |
| Dual | Separate models | Error difference | Interpretability |

Results are saved to `results/` directory with detailed per-dataset breakdowns.

Part of the implementation refers to their [original repository](https://github.com/ShenzheZhu/JailDAM). If by any chance this repository is useful to you, you only need to cite their work:

```bibtex
@article{nian2025jaildam,
  title={JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model},
  author={Nian, Yi and Zhu, Shenzhe and Qin, Yuehan and Li, Li and Wang, Ziyi and Xiao, Chaowei and Zhao, Yue},
  journal={arXiv preprint arXiv:2504.03770},
  year={2025}
}
```
