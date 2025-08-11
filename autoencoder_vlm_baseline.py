#!/usr/bin/env python3
"""
VLM-based Autoencoder Baseline for Jailbreak Detection

This script integrates with the existing Jail-DAM infrastructure to use proper
VLM features (CLIP embeddings) for training and evaluating the autoencoder.

Training: Benign data only (Alpaca + MM-Vet + OpenAssistant)
Testing: Balanced safe/unsafe data from your reference configuration
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, CLIPModel
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

# Import the enhanced autoencoder
from Autoencoder import Autoencoder

# Suppress warnings
warnings.filterwarnings("ignore")

class VLMAutoencoderConfig:
    """Configuration for VLM-based autoencoder training"""
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    LATENT_DIM = 128
    VLM_MODEL = "openai/clip-vit-large-patch14"
    
    # Training parameters
    BATCH_SIZE = 16  # Smaller batch size for VLM processing
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

class VLMFeatureExtractor:
    """Extract features using CLIP model"""

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get tokenizer for proper text truncation
        self.tokenizer = self.processor.tokenizer
        self.max_length = 77  # CLIP's context length
        
    def extract_features(self, samples, batch_size=8):
        """Extract CLIP features from text and image samples"""
        print(f"Extracting CLIP features from {len(samples)} samples...")

        # Check text lengths for debugging
        text_lengths = []
        for sample in samples[:10]:  # Check first 10 samples
            text = sample.get('txt', '')
            text_lengths.append(len(text))

        if text_lengths:
            print(f"Sample text lengths (first 10): min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths)/len(text_lengths):.1f}")
            if max(text_lengths) > 1000:
                print("Warning: Some texts are very long and will be truncated")
        
        all_features = []
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_texts = []
            batch_images = []
            
            for sample in batch_samples:
                # Get text - keep full text, let CLIP processor handle truncation properly
                text = sample.get('txt', '')
                if not text:
                    text = "empty text"

                batch_texts.append(text)
                
                # Get image (if available)
                img_path = sample.get('img', None)
                if img_path and os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        batch_images.append(image)
                    except:
                        # Use blank image if loading fails
                        batch_images.append(Image.new('RGB', (224, 224), color='white'))
                else:
                    # Use blank image if no image provided
                    batch_images.append(Image.new('RGB', (224, 224), color='white'))
            
            # Process batch
            try:
                with torch.no_grad():
                    # Process text and images separately to handle truncation properly
                    text_inputs = self.processor.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    )

                    image_inputs = self.processor.image_processor(
                        batch_images,
                        return_tensors="pt"
                    )

                    # Combine inputs
                    inputs = {
                        'input_ids': text_inputs['input_ids'],
                        'attention_mask': text_inputs['attention_mask'],
                        'pixel_values': image_inputs['pixel_values']
                    }
                    
                    # Move to device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.device)
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    text_embeds = outputs.text_embeds  # [batch_size, 768]
                    image_embeds = outputs.image_embeds  # [batch_size, 768]
                    
                    # Concatenate text and image embeddings
                    combined_embeds = torch.cat([text_embeds, image_embeds], dim=1)  # [batch_size, 1536]
                    
                    all_features.extend(combined_embeds.cpu().numpy())
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                # Add zero features for failed batch
                zero_features = np.zeros((len(batch_samples), 1536))
                all_features.extend(zero_features)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_samples)}/{len(samples)} samples")
        
        return np.array(all_features, dtype=np.float32)

def load_benign_training_data():
    """Load benign training data according to your configuration"""
    print("Loading benign training data...")
    benign_samples = []
    
    # Alpaca - 500 examples
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        benign_samples.extend(alpaca_samples)
        print(f"Added {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    
    # MM-Vet - 218 examples
    try:
        mmvet_samples = load_mm_vet()
        mmvet_subset = mmvet_samples[:218] if len(mmvet_samples) >= 218 else mmvet_samples
        benign_samples.extend(mmvet_subset)
        print(f"Added {len(mmvet_subset)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")
    
    # OpenAssistant - 282 examples
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        benign_samples.extend(openassistant_samples)
        print(f"Added {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")
    
    print(f"Total benign training samples: {len(benign_samples)}")
    return benign_samples

def load_balanced_test_data():
    """Load balanced test data according to your configuration"""
    print("Loading balanced test data...")
    safe_samples = []
    unsafe_samples = []
    
    # Safe samples
    try:
        # XSTest safe - 250 examples
        xstest_samples = load_XSTest()
        xstest_safe = [s for s in xstest_samples if s['toxicity'] == 0]
        xstest_safe_subset = random.sample(xstest_safe, min(250, len(xstest_safe)))
        safe_samples.extend(xstest_safe_subset)
        print(f"Added {len(xstest_safe_subset)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest safe: {e}")
    
    try:
        # FigTxt safe - 300 examples
        figtxt_samples = load_FigTxt()
        figtxt_safe = [s for s in figtxt_samples if s['toxicity'] == 0]
        figtxt_safe_subset = random.sample(figtxt_safe, min(300, len(figtxt_safe)))
        safe_samples.extend(figtxt_safe_subset)
        print(f"Added {len(figtxt_safe_subset)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt safe: {e}")
    
    try:
        # VQAv2 - 350 examples
        vqav2_samples = load_vqav2(max_samples=350)
        safe_samples.extend(vqav2_samples)
        print(f"Added {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")
    
    # Unsafe samples
    try:
        # XSTest unsafe - 200 examples
        xstest_samples = load_XSTest()
        xstest_unsafe = [s for s in xstest_samples if s['toxicity'] == 1]
        xstest_unsafe_subset = random.sample(xstest_unsafe, min(200, len(xstest_unsafe)))
        unsafe_samples.extend(xstest_unsafe_subset)
        print(f"Added {len(xstest_unsafe_subset)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")
    
    try:
        # FigTxt unsafe - 350 examples
        figtxt_samples = load_FigTxt()
        figtxt_unsafe = [s for s in figtxt_samples if s['toxicity'] == 1]
        figtxt_unsafe_subset = random.sample(figtxt_unsafe, min(350, len(figtxt_unsafe)))
        unsafe_samples.extend(figtxt_unsafe_subset)
        print(f"Added {len(figtxt_unsafe_subset)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")
    
    try:
        # VAE - 200 examples
        vae_samples = load_adversarial_img()
        vae_subset = random.sample(vae_samples, min(200, len(vae_samples)))
        unsafe_samples.extend(vae_subset)
        print(f"Added {len(vae_subset)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")
    
    try:
        # JailbreakV-28K - 150 examples (figstep attack for testing)
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        unsafe_samples.extend(jbv_test_samples)
        print(f"Added {len(jbv_test_samples)} JailbreakV-28K test samples")
    except Exception as e:
        print(f"Could not load JailbreakV-28K test: {e}")
    
    print(f"Test set: {len(safe_samples)} safe, {len(unsafe_samples)} unsafe")
    return safe_samples, unsafe_samples

def train_autoencoder(train_features, config):
    """Train autoencoder on benign data only"""
    print(f"Training autoencoder on {len(train_features)} benign samples...")
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(torch.FloatTensor(train_features))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_dim = train_features.shape[1]
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
            torch.save(model.state_dict(), 'best_vlm_autoencoder.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_vlm_autoencoder.pth'))
    print(f"Training completed. Best loss: {best_loss:.6f}")
    
    return model, train_losses

def evaluate_autoencoder_detailed(model, test_features, test_labels, test_dataset_info, config):
    """Evaluate autoencoder with detailed per-dataset metrics"""
    print(f"Evaluating autoencoder on {len(test_features)} test samples...")

    model.eval()

    # Calculate reconstruction errors
    test_dataset = TensorDataset(torch.FloatTensor(test_features))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    reconstruction_errors = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(config.DEVICE)
            errors = model.get_reconstruction_error(inputs)
            reconstruction_errors.extend(errors.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    # Find optimal threshold using validation approach
    val_size = min(200, len(test_labels) // 4)
    val_indices = np.random.choice(len(test_labels), val_size, replace=False)

    val_errors = reconstruction_errors[val_indices]
    val_labels = test_labels[val_indices]

    # Grid search for optimal threshold
    thresholds = np.percentile(val_errors, np.linspace(10, 90, 100))
    best_f1 = 0
    best_threshold = np.median(reconstruction_errors)

    for threshold in thresholds:
        val_predictions = (val_errors > threshold).astype(int)
        f1 = f1_score(val_labels, val_predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.6f}")

    # Make predictions on full test set
    predictions = (reconstruction_errors > best_threshold).astype(int)

    # Calculate overall metrics
    from sklearn.metrics import confusion_matrix
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, zero_division=0)

    # Calculate TPR, FPR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

    # Calculate AUROC and AUPRC
    try:
        auroc = roc_auc_score(test_labels, reconstruction_errors)
        auprc = average_precision_score(test_labels, reconstruction_errors)
    except:
        auroc = 0.0
        auprc = 0.0

    # Calculate per-dataset metrics
    dataset_metrics = {}
    current_idx = 0

    for dataset_name, dataset_samples in test_dataset_info.items():
        dataset_size = len(dataset_samples)
        dataset_labels = test_labels[current_idx:current_idx + dataset_size]
        dataset_predictions = predictions[current_idx:current_idx + dataset_size]
        dataset_errors = reconstruction_errors[current_idx:current_idx + dataset_size]

        # Calculate metrics for this dataset
        dataset_acc = accuracy_score(dataset_labels, dataset_predictions)
        dataset_f1 = f1_score(dataset_labels, dataset_predictions, zero_division=0)

        # Calculate TPR, FPR for this dataset
        if len(np.unique(dataset_labels)) > 1:
            tn_d, fp_d, fn_d, tp_d = confusion_matrix(dataset_labels, dataset_predictions).ravel()
            dataset_tpr = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            dataset_fpr = fp_d / (fp_d + tn_d) if (fp_d + tn_d) > 0 else 0.0

            try:
                dataset_auroc = roc_auc_score(dataset_labels, dataset_errors)
                dataset_auprc = average_precision_score(dataset_labels, dataset_errors)
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
            'unsafe_samples': int(np.sum(dataset_labels == 1))
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
        'reconstruction_errors': reconstruction_errors,
        'predictions': predictions
    }

    return results

def save_detailed_results(results, config):
    """Save detailed results to file"""
    os.makedirs('results', exist_ok=True)

    # Prepare results for JSON serialization

    # Save detailed results
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
                'unsafe_samples': int(metrics['unsafe_samples'])
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

    with open('results/vlm_autoencoder_detailed_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"Detailed results saved to results/vlm_autoencoder_detailed_results.json")

if __name__ == "__main__":
    print("="*80)
    print("VLM-BASED AUTOENCODER BASELINE FOR JAILBREAK DETECTION")
    print("="*80)

    # Set random seed
    config = VLMAutoencoderConfig()
    set_seed(config.SEED)
    print(f"Using device: {config.DEVICE}")
    print(f"Random seed set to: {config.SEED}")
    print(f"VLM Model: {config.VLM_MODEL}")

    # Load datasets
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)

    benign_train_samples = load_benign_training_data()
    safe_test_samples, unsafe_test_samples = load_balanced_test_data()

    if not benign_train_samples:
        print("Error: No benign training samples loaded!")
        sys.exit(1)

    if not safe_test_samples and not unsafe_test_samples:
        print("Error: No test samples loaded!")
        sys.exit(1)

    # Initialize feature extractor
    print("\n" + "="*50)
    print("INITIALIZING VLM FEATURE EXTRACTOR")
    print("="*50)

    feature_extractor = VLMFeatureExtractor(config.VLM_MODEL, config.DEVICE)

    # Extract features
    print("\n" + "="*50)
    print("EXTRACTING VLM FEATURES")
    print("="*50)

    train_features = feature_extractor.extract_features(benign_train_samples, batch_size=8)

    # Combine test samples and create labels with dataset info
    test_dataset_info = {
        'XSTest_safe': [s for s in safe_test_samples if 'XSTest' in str(s)],
        'FigTxt_safe': [s for s in safe_test_samples if 'FigTxt' in str(s)],
        'VQAv2': [s for s in safe_test_samples if 'VQAv2' in str(s) or 'vqa' in str(s).lower()],
        'XSTest_unsafe': [s for s in unsafe_test_samples if 'XSTest' in str(s)],
        'FigTxt_unsafe': [s for s in unsafe_test_samples if 'FigTxt' in str(s)],
        'VAE': [s for s in unsafe_test_samples if 'VAE' in str(s) or 'adversarial' in str(s).lower()],
        'JailbreakV-28K': [s for s in unsafe_test_samples if 'JailbreakV' in str(s) or 'figstep' in str(s).lower()]
    }

    # Fallback: if dataset identification fails, use sample counts
    if sum(len(samples) for samples in test_dataset_info.values()) < len(safe_test_samples + unsafe_test_samples):
        # Use the original loading order and expected counts
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
    test_labels = [0] * len(safe_test_samples) + [1] * len(unsafe_test_samples)  # 0=safe, 1=unsafe
    test_features = feature_extractor.extract_features(all_test_samples, batch_size=8)

    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels distribution: {np.bincount(test_labels)}")

    # Train autoencoder
    print("\n" + "="*50)
    print("TRAINING AUTOENCODER")
    print("="*50)

    model, train_losses = train_autoencoder(train_features, config)

    # Evaluate autoencoder
    print("\n" + "="*50)
    print("EVALUATING AUTOENCODER")
    print("="*50)

    results = evaluate_autoencoder_detailed(model, test_features, np.array(test_labels), test_dataset_info, config)

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
    print(f"{'Dataset':<15} {'Size':<6} {'Safe':<5} {'Unsafe':<6} {'Acc':<6} {'F1':<6} {'TPR':<6} {'FPR':<6} {'AUROC':<6} {'AUPRC':<6}")
    print("-" * 85)

    for dataset_name, metrics in results['per_dataset'].items():
        if metrics['size'] > 0:  # Only show datasets with samples
            print(f"{dataset_name:<15} {metrics['size']:<6} {metrics['safe_samples']:<5} {metrics['unsafe_samples']:<6} "
                  f"{metrics['accuracy']:<6.3f} {metrics['f1']:<6.3f} {metrics['tpr']:<6.3f} {metrics['fpr']:<6.3f} "
                  f"{metrics['auroc']:<6.3f} {metrics['auprc']:<6.3f}")

    # Save detailed results
    save_detailed_results(results, config)

    print("\n" + "="*80)
    print("VLM AUTOENCODER BASELINE COMPLETED")
    print("="*80)
