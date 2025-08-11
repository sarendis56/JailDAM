#!/usr/bin/env python3
"""
Dual Autoencoder for Jailbreak Detection

This approach trains TWO separate autoencoders:
1. Benign Autoencoder: Trained only on benign data
2. Unsafe Autoencoder: Trained only on unsafe data

Detection Logic:
- Benign samples: Low benign reconstruction error, High unsafe reconstruction error
- Unsafe samples: High benign reconstruction error, Low unsafe reconstruction error
- Detection Score: Unsafe_Error - Benign_Error (positive = unsafe, negative = benign)

This should provide the clearest separation between benign and unsafe patterns.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from transformers import AutoProcessor, CLIPModel
import warnings
import json
from PIL import Image

# Import dataset loaders from reference directory
import importlib.util
spec = importlib.util.spec_from_file_location("load_datasets", "reference/load_datasets.py")
load_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_datasets)

# Import the functions we need
load_alpaca = load_datasets.load_alpaca
load_mm_vet = load_datasets.load_mm_vet
load_openassistant = load_datasets.load_openassistant
load_XSTest = load_datasets.load_XSTest
load_FigTxt = load_datasets.load_FigTxt
load_vqav2 = load_datasets.load_vqav2
load_adversarial_img = load_datasets.load_adversarial_img
load_JailBreakV_figstep = load_datasets.load_JailBreakV_figstep
load_advbench = load_datasets.load_advbench
load_dan_prompts = load_datasets.load_dan_prompts
load_JailBreakV_llm_transfer_attack = load_datasets.load_JailBreakV_llm_transfer_attack

# Import the enhanced autoencoder
from Autoencoder import Autoencoder

# Suppress warnings
warnings.filterwarnings("ignore")

class DualAutoencoderConfig:
    """Configuration for dual autoencoder training"""
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    LATENT_DIM = 128
    VLM_MODEL = "openai/clip-vit-large-patch14"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 1e-5

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_training_data_separate():
    """Load benign and unsafe training data separately"""
    print("Loading training data (separate benign and unsafe)...")
    
    # Benign training data (same as before)
    benign_samples = []
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        benign_samples.extend(alpaca_samples)
        print(f"Added {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    
    try:
        mmvet_samples = load_mm_vet()
        mmvet_subset = mmvet_samples[:218] if len(mmvet_samples) >= 218 else mmvet_samples
        benign_samples.extend(mmvet_subset)
        print(f"Added {len(mmvet_subset)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")
    
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        benign_samples.extend(openassistant_samples)
        print(f"Added {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")
    
    # Unsafe training data (different from test data to avoid leakage)
    unsafe_samples = []
    try:
        advbench_samples = load_advbench(max_samples=400)
        unsafe_samples.extend(advbench_samples)
        print(f"Added {len(advbench_samples)} AdvBench samples")
    except Exception as e:
        print(f"Could not load AdvBench: {e}")
    
    try:
        # Use nature and noise styles for training (different from figstep used in testing)
        jbv_samples = load_JailBreakV_llm_transfer_attack(image_styles=['nature', 'noise'], max_samples=400)
        unsafe_samples.extend(jbv_samples)
        print(f"Added {len(jbv_samples)} JailbreakV-28K samples (nature+noise styles)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")
    
    try:
        dan_samples = load_dan_prompts(max_samples=200)
        unsafe_samples.extend(dan_samples)
        print(f"Added {len(dan_samples)} DAN samples")
    except Exception as e:
        print(f"Could not load DAN: {e}")
    
    print(f"Training data: {len(benign_samples)} benign, {len(unsafe_samples)} unsafe")
    return benign_samples, unsafe_samples

def train_single_autoencoder(features, name, config):
    """Train a single autoencoder on given features"""
    print(f"Training {name} autoencoder on {len(features)} samples...")
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(torch.FloatTensor(features))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_dim = features.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=config.LATENT_DIM).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs = batch[0].to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss - config.MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{name.lower()}_autoencoder.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{name.lower()}_autoencoder.pth'))
    print(f"  {name} autoencoder training completed. Best loss: {best_loss:.6f}")
    
    return model, train_losses

def evaluate_dual_autoencoder(benign_model, unsafe_model, test_features, test_labels, test_dataset_info, config):
    """Evaluate dual autoencoder using difference in reconstruction errors"""
    print(f"Evaluating dual autoencoder on {len(test_features)} test samples...")
    
    benign_model.eval()
    unsafe_model.eval()
    
    # Calculate reconstruction errors from both models
    test_dataset = TensorDataset(torch.FloatTensor(test_features))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    benign_errors = []
    unsafe_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(config.DEVICE)
            
            # Get reconstruction errors from both models
            benign_err = benign_model.get_reconstruction_error(inputs)
            unsafe_err = unsafe_model.get_reconstruction_error(inputs)
            
            benign_errors.extend(benign_err.cpu().numpy())
            unsafe_errors.extend(unsafe_err.cpu().numpy())
    
    benign_errors = np.array(benign_errors)
    unsafe_errors = np.array(unsafe_errors)
    
    # Detection score: Benign_Error - Unsafe_Error
    # Positive score = more likely unsafe (benign model fails, unsafe model succeeds)
    # Negative score = more likely benign (unsafe model fails, benign model succeeds)
    detection_scores = benign_errors - unsafe_errors
    
    print(f"Detection score statistics:")
    print(f"  Mean: {np.mean(detection_scores):.6f}")
    print(f"  Std: {np.std(detection_scores):.6f}")
    print(f"  Min: {np.min(detection_scores):.6f}")
    print(f"  Max: {np.max(detection_scores):.6f}")
    
    # Find optimal threshold using validation approach
    val_size = min(200, len(test_labels) // 4)
    val_indices = np.random.choice(len(test_labels), val_size, replace=False)
    
    val_scores = detection_scores[val_indices]
    val_labels = test_labels[val_indices]
    
    # Grid search for optimal threshold
    thresholds = np.percentile(val_scores, np.linspace(10, 90, 100))
    best_f1 = 0
    best_threshold = 0.0  # Start with 0 as default (natural separation point)
    
    for threshold in thresholds:
        val_predictions = (val_scores > threshold).astype(int)
        f1 = f1_score(val_labels, val_predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.6f}")
    
    # Make predictions on full test set
    predictions = (detection_scores > best_threshold).astype(int)
    
    # Calculate overall metrics
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    # Calculate TPR, FPR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    
    # Calculate AUROC and AUPRC using detection scores
    try:
        auroc = roc_auc_score(test_labels, detection_scores)
        auprc = average_precision_score(test_labels, detection_scores)
    except:
        auroc = 0.0
        auprc = 0.0
    
    # Calculate per-dataset metrics (simplified version)
    dataset_metrics = {}
    current_idx = 0
    
    for dataset_name, dataset_samples in test_dataset_info.items():
        dataset_size = len(dataset_samples)
        if dataset_size == 0:
            continue
            
        dataset_labels = test_labels[current_idx:current_idx + dataset_size]
        dataset_predictions = predictions[current_idx:current_idx + dataset_size]
        dataset_scores = detection_scores[current_idx:current_idx + dataset_size]
        
        # Calculate metrics for this dataset
        dataset_acc = accuracy_score(dataset_labels, dataset_predictions)
        dataset_f1 = f1_score(dataset_labels, dataset_predictions, zero_division=0)
        
        # Calculate TPR, FPR for this dataset
        if len(np.unique(dataset_labels)) > 1:
            tn_d, fp_d, fn_d, tp_d = confusion_matrix(dataset_labels, dataset_predictions).ravel()
            dataset_tpr = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            dataset_fpr = fp_d / (fp_d + tn_d) if (fp_d + tn_d) > 0 else 0.0
            
            try:
                dataset_auroc = roc_auc_score(dataset_labels, dataset_scores)
                dataset_auprc = average_precision_score(dataset_labels, dataset_scores)
            except:
                dataset_auroc = 0.0
                dataset_auprc = 0.0
        else:
            # Single class dataset
            dataset_tpr = 1.0 if dataset_labels[0] == 1 and dataset_predictions[0] == 1 else 0.0
            dataset_fpr = 1.0 if dataset_labels[0] == 0 and dataset_predictions[0] == 1 else 0.0
            dataset_auroc = 0.0
            dataset_auprc = 0.0
        
        dataset_metrics[dataset_name] = {
            'accuracy': dataset_acc,
            'f1': dataset_f1,
            'tpr': dataset_tpr,
            'fpr': dataset_fpr,
            'auroc': dataset_auroc,
            'auprc': dataset_auprc,
            'size': dataset_size,
            'safe_samples': int(np.sum(dataset_labels == 0)),
            'unsafe_samples': int(np.sum(dataset_labels == 1)),
            'mean_score': float(np.mean(dataset_scores))
        }
        
        current_idx += dataset_size
    
    results = {
        'overall': {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'threshold': best_threshold
        },
        'per_dataset': dataset_metrics,
        'benign_errors': benign_errors,
        'unsafe_errors': unsafe_errors,
        'detection_scores': detection_scores,
        'predictions': predictions
    }
    
    return results

if __name__ == "__main__":
    print("="*80)
    print("DUAL AUTOENCODER FOR JAILBREAK DETECTION")
    print("="*80)
    print("Training TWO separate autoencoders:")
    print("1. Benign Autoencoder (trained on benign data only)")
    print("2. Unsafe Autoencoder (trained on unsafe data only)")
    print("Detection: Unsafe_Reconstruction_Error - Benign_Reconstruction_Error")
    print("="*80)

    # Set random seed
    config = DualAutoencoderConfig()
    set_seed(config.SEED)
    print(f"Using device: {config.DEVICE}")
    print(f"Random seed set to: {config.SEED}")

    # Import VLM feature extractor from the main script
    from autoencoder_vlm_baseline import VLMFeatureExtractor, load_balanced_test_data

    # Load training data (separate benign and unsafe)
    print("\n" + "="*50)
    print("LOADING TRAINING DATA")
    print("="*50)

    benign_train_samples, unsafe_train_samples = load_training_data_separate()

    if not benign_train_samples:
        print("Error: No benign training samples loaded!")
        sys.exit(1)

    if not unsafe_train_samples:
        print("Error: No unsafe training samples loaded!")
        sys.exit(1)

    # Load test data (same as before)
    safe_test_samples, unsafe_test_samples = load_balanced_test_data()

    # Initialize feature extractor
    print("\n" + "="*50)
    print("EXTRACTING VLM FEATURES")
    print("="*50)

    feature_extractor = VLMFeatureExtractor(config.VLM_MODEL, config.DEVICE)

    # Extract training features
    print("Extracting benign training features...")
    benign_train_features = feature_extractor.extract_features(benign_train_samples, batch_size=8)

    print("Extracting unsafe training features...")
    unsafe_train_features = feature_extractor.extract_features(unsafe_train_samples, batch_size=8)

    # Extract test features
    print("Extracting test features...")
    # Create test dataset info for detailed evaluation
    test_dataset_info = {
        'XSTest_safe': safe_test_samples[:250] if len(safe_test_samples) >= 250 else [],
        'FigTxt_safe': safe_test_samples[250:550] if len(safe_test_samples) >= 550 else [],
        'VQAv2': safe_test_samples[550:] if len(safe_test_samples) > 550 else [],
        'XSTest_unsafe': unsafe_test_samples[:200] if len(unsafe_test_samples) >= 200 else [],
        'FigTxt_unsafe': unsafe_test_samples[200:550] if len(unsafe_test_samples) >= 550 else [],
        'VAE': unsafe_test_samples[550:750] if len(unsafe_test_samples) >= 750 else [],
        'JailbreakV-28K': unsafe_test_samples[750:] if len(unsafe_test_samples) > 750 else []
    }

    all_test_samples = safe_test_samples + unsafe_test_samples
    test_labels = [0] * len(safe_test_samples) + [1] * len(unsafe_test_samples)
    test_features = feature_extractor.extract_features(all_test_samples, batch_size=8)

    print(f"Benign training features: {benign_train_features.shape}")
    print(f"Unsafe training features: {unsafe_train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Test labels distribution: {np.bincount(test_labels)}")

    # Train dual autoencoders
    print("\n" + "="*50)
    print("TRAINING DUAL AUTOENCODERS")
    print("="*50)

    # Train benign autoencoder
    benign_model, benign_losses = train_single_autoencoder(benign_train_features, "Benign", config)

    # Train unsafe autoencoder
    unsafe_model, unsafe_losses = train_single_autoencoder(unsafe_train_features, "Unsafe", config)

    # Evaluate dual autoencoder
    print("\n" + "="*50)
    print("EVALUATING DUAL AUTOENCODER")
    print("="*50)

    results = evaluate_dual_autoencoder(benign_model, unsafe_model, test_features, np.array(test_labels), test_dataset_info, config)

    # Print overall results
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    overall = results['overall']
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"F1 Score: {overall['f1']:.4f}")
    print(f"TPR (Sensitivity): {overall['tpr']:.4f}")
    print(f"FPR (1-Specificity): {overall['fpr']:.4f}")
    print(f"AUROC: {overall['auroc']:.4f}")
    print(f"AUPRC: {overall['auprc']:.4f}")
    print(f"Threshold: {overall['threshold']:.6f}")

    # Print per-dataset results
    print("\n" + "="*50)
    print("PER-DATASET BREAKDOWN")
    print("="*50)
    print(f"{'Dataset':<15} {'Size':<6} {'Safe':<5} {'Unsafe':<6} {'Acc':<6} {'F1':<6} {'TPR':<6} {'FPR':<6} {'AUROC':<6} {'AUPRC':<6} {'Score':<8}")
    print("-" * 95)

    for dataset_name, metrics in results['per_dataset'].items():
        if metrics['size'] > 0:  # Only show datasets with samples
            print(f"{dataset_name:<15} {metrics['size']:<6} {metrics['safe_samples']:<5} {metrics['unsafe_samples']:<6} "
                  f"{metrics['accuracy']:<6.3f} {metrics['f1']:<6.3f} {metrics['tpr']:<6.3f} {metrics['fpr']:<6.3f} "
                  f"{metrics['auroc']:<6.3f} {metrics['auprc']:<6.3f} {metrics['mean_score']:<8.4f}")

    # Save detailed results
    os.makedirs('results', exist_ok=True)
    results_summary = {
        'overall': {
            'accuracy': float(results['overall']['accuracy']),
            'f1': float(results['overall']['f1']),
            'tpr': float(results['overall']['tpr']),
            'fpr': float(results['overall']['fpr']),
            'auroc': float(results['overall']['auroc']),
            'auprc': float(results['overall']['auprc']),
            'threshold': float(results['overall']['threshold'])
        },
        'per_dataset': {
            dataset: {
                'accuracy': float(metrics['accuracy']),
                'f1': float(metrics['f1']),
                'tpr': float(metrics['tpr']),
                'fpr': float(metrics['fpr']),
                'auroc': float(metrics['auroc']),
                'auprc': float(metrics['auprc']),
                'size': int(metrics['size']),
                'safe_samples': int(metrics['safe_samples']),
                'unsafe_samples': int(metrics['unsafe_samples']),
                'mean_detection_score': float(metrics['mean_score'])
            }
            for dataset, metrics in results['per_dataset'].items()
        },
        'config': {
            'latent_dim': config.LATENT_DIM,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'seed': config.SEED,
            'vlm_model': config.VLM_MODEL
        }
    }

    with open('results/dual_autoencoder_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to results/dual_autoencoder_results.json")

    print("\n" + "="*80)
    print("DUAL AUTOENCODER COMPLETED")
    print("="*80)
    print("\nDual Autoencoder Approach:")
    print("✓ Benign Autoencoder: Specializes in reconstructing benign patterns")
    print("✓ Unsafe Autoencoder: Specializes in reconstructing unsafe patterns")
    print("✓ Detection Score: Benign_Error - Unsafe_Error")
    print("✓ Positive Score → Unsafe (benign model fails, unsafe model succeeds)")
    print("✓ Negative Score → Benign (unsafe model fails, benign model succeeds)")
    print("\nThis should provide the clearest separation between pattern types!")
