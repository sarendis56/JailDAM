#!/usr/bin/env python3
"""
MM-Vet Only Training Experiment for Jailbreak Detection

This script investigates whether training the autoencoder on a single benign dataset
(MM-Vet) leads to rejection of other benign datasets with different distributions.

Training: MM-Vet only (multimodal benign)
Testing: 
- Benign: SciQ, TruthfulQA, VCR, ScienceQA (from HuggingFace)
- Malicious: JailBreakV-28k subset (local)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from transformers import AutoProcessor, CLIPModel
import warnings
import json
from PIL import Image

# Import dataset loaders
import importlib.util
spec = importlib.util.spec_from_file_location("load_datasets", "reference/load_datasets.py")
load_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_datasets)

# Import HuggingFace dataset loaders
from load_hf_datasets import load_all_test_datasets, load_jailbreakv_subset

# Import the enhanced autoencoder
from Autoencoder import Autoencoder

# Suppress warnings
warnings.filterwarnings("ignore")

class MMVetOnlyConfig:
    """Configuration for MM-Vet only training experiment"""
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    LATENT_DIM = 64  # Official JailDAM uses 64
    VLM_MODEL = "openai/clip-vit-large-patch14"
    
    # Official JailDAM parameters
    NUM_CATEGORIES_TRAIN = 8
    SAMPLES_PER_CATEGORY = 100
    AUTOENCODER_INPUT_DIM = NUM_CATEGORIES_TRAIN * SAMPLES_PER_CATEGORY * 2  # 1600
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 1e-5
    
    # Test dataset limits
    MAX_SAMPLES_PER_DATASET = 300  # Balanced samples for benign datasets
    MAX_JAILBREAK_SAMPLES = 2000  # Load up to 2000 JailBreakV samples

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class VLMFeatureExtractor:
    """VLM feature extractor using CLIP"""
    
    def __init__(self, model_name, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def extract_features(self, samples, batch_size=8):
        """Extract CLIP features from samples"""
        features = []
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_features = self._extract_batch_features(batch_samples)
            features.append(batch_features)
        
        return np.vstack(features)
    
    def _extract_batch_features(self, samples):
        """Extract features for a batch of samples"""
        texts = []
        images = []
        
        for sample in samples:
            texts.append(sample['txt'])
            
            # Handle image
            if sample.get('img') and os.path.exists(sample['img']):
                try:
                    image = Image.open(sample['img']).convert('RGB')
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {sample['img']}: {e}")
                    images.append(None)
            else:
                if sample.get('img'):
                    print(f"Image file not found: {sample['img']}")
                images.append(None)
        # Check if we have any valid images
        has_images = any(img is not None for img in images)
        
        # Debug: Check sample types
        text_only_count = sum(1 for img in images if img is None)
        multimodal_count = len(images) - text_only_count
        if text_only_count > 0 and multimodal_count > 0:
            print(f"  Mixed batch: {text_only_count} text-only, {multimodal_count} multimodal")
        
        with torch.no_grad():
            if has_images:
                # Mixed case - process each sample individually to handle None images
                features_list = []
                for i, (text, img) in enumerate(zip(texts, images)):
                    if img is not None:
                        # Process with image
                        inputs = self.processor(
                            text=[text],
                            images=[img],
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                        
                        outputs = self.model(**inputs)
                        text_feat = outputs.text_embeds
                        img_feat = outputs.image_embeds
                        sample_features = torch.cat([text_feat, img_feat], dim=1)
                    else:
                        # Process text-only
                        inputs = self.processor(
                            text=[text],
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                        
                        text_inputs = {
                            'input_ids': inputs['input_ids'],
                            'attention_mask': inputs['attention_mask']
                        }
                        
                        text_outputs = self.model.get_text_features(**text_inputs)
                        # Duplicate text features to match multimodal dimension
                        sample_features = torch.cat([text_outputs, text_outputs], dim=1)
                    
                    features_list.append(sample_features)
                
                features = torch.cat(features_list, dim=0)
            else:
                # Text-only case - use only text processing
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # For text-only, we need to explicitly avoid vision processing
                # by only passing text-related inputs to the model
                text_inputs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
                
                # Get text features only
                text_outputs = self.model.get_text_features(**text_inputs)
                # Concatenate text features twice to match multimodal feature dimension
                # MM-Vet features are 1536 (768 text + 768 image), so we duplicate text to 1536
                text_features = text_outputs  # Shape: [batch_size, 768]
                features = torch.cat([text_features, text_features], dim=1)  # Shape: [batch_size, 1536]
        
        return features.cpu().numpy()

def load_mmvet_data():
    """Load MM-Vet dataset and split into training and testing"""
    print("Loading MM-Vet dataset...")
    
    # Load directly from sample.json with custom parsing
    import json
    
    with open("data/mm-vet/sample.json", "r") as f:
        sample_data = json.load(f)
    
    mmvet_samples = []
    for item in sample_data:
        qid = item["id"]
        question = item["question"]
        image_path = item["image_path"]
        
        # Convert relative path to absolute path
        if image_path.startswith("./"):
            image_path = f"data/mm-vet/{image_path[2:]}"
        
        sample = {
            "txt": question,
            "img": image_path,
            "toxicity": 0,  # MM-Vet is benign
            "question_id": qid
        }
        mmvet_samples.append(sample)
    
    print(f"Loaded {len(mmvet_samples)} MM-Vet samples")
    
    # Split into 80% training, 20% testing
    set_seed(42)  # Ensure reproducible split
    random.shuffle(mmvet_samples)
    
    split_idx = int(0.8 * len(mmvet_samples))
    train_samples = mmvet_samples[:split_idx]
    test_samples = mmvet_samples[split_idx:]
    
    print(f"Split: {len(train_samples)} training samples, {len(test_samples)} testing samples")
    
    return train_samples, test_samples

def generate_concept_embeddings(samples, config):
    """Generate concept embeddings from samples (simplified version)"""
    print("Generating concept embeddings...")
    
    # For simplicity, we'll create random concept embeddings
    # In the official implementation, these are learned from the data
    num_concepts = config.NUM_CATEGORIES_TRAIN * config.SAMPLES_PER_CATEGORY
    concept_embeddings = torch.randn(num_concepts, 1536, device=config.DEVICE) * 0.1
    
    print(f"Generated {num_concepts} concept embeddings with shape {concept_embeddings.shape}")
    return concept_embeddings

def compute_attention_features(text_embeddings, image_embeddings, concept_embeddings):
    """Compute attention features using concept embeddings (official JailDAM approach)"""
    # Split concept embeddings for text and image
    text_concepts = concept_embeddings[:, :768]  # First 768 dimensions for text
    image_concepts = concept_embeddings[:, 768:]  # Last 768 dimensions for image
    
    # Compute similarities
    sim_txt = text_embeddings @ text_concepts.T  # [batch, num_concepts]
    sim_img = image_embeddings @ image_concepts.T  # [batch, num_concepts]
    
    # Concatenate similarities to create attention features
    attention_features = torch.cat((sim_txt, sim_img), dim=-1)  # [batch, 2*num_concepts]
    
    return attention_features

def train_autoencoder_on_mmvet(features, concept_embeddings, config):
    """Train autoencoder on MM-Vet attention features (official JailDAM approach)"""
    print(f"Training autoencoder on {len(features)} MM-Vet samples...")
    
    # Convert CLIP features to attention features
    print("Converting CLIP features to attention features...")
    attention_features_list = []
    
    for i in range(0, len(features), config.BATCH_SIZE):
        batch_features = features[i:i+config.BATCH_SIZE]
        batch_text = batch_features[:, :768]  # First 768 dimensions (text)
        batch_image = batch_features[:, 768:]  # Last 768 dimensions (image)
        
        # Compute attention features
        attention_batch = compute_attention_features(
            torch.FloatTensor(batch_text).to(config.DEVICE),
            torch.FloatTensor(batch_image).to(config.DEVICE),
            concept_embeddings
        )
        attention_features_list.append(attention_batch.cpu())
    
    attention_features = torch.cat(attention_features_list, dim=0).numpy()
    print(f"Attention features shape: {attention_features.shape}")
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(torch.FloatTensor(attention_features))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Initialize model with official JailDAM architecture
    model = Autoencoder(input_dim=config.AUTOENCODER_INPUT_DIM, latent_dim=config.LATENT_DIM).to(config.DEVICE)
    
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
            torch.save(model.state_dict(), 'best_mmvet_autoencoder.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_mmvet_autoencoder.pth'))
    print(f"  MM-Vet autoencoder training completed. Best loss: {best_loss:.6f}")
    
    return model, train_losses

def evaluate_on_test_datasets(model, feature_extractor, test_data, concept_embeddings, config):
    """Evaluate model on all test datasets using attention features (official JailDAM approach)"""
    print("Evaluating on test datasets...")
    
    model.eval()
    results = {}
    
    # Process each test dataset
    for dataset_name, samples in test_data.items():
        if not samples:
            continue
            
        print(f"\nEvaluating on {dataset_name} ({len(samples)} samples)...")
        
        # Extract CLIP features
        clip_features = feature_extractor.extract_features(samples, batch_size=8)
        
        # Convert to attention features
        print("Converting to attention features...")
        attention_features_list = []
        
        for i in range(0, len(clip_features), config.BATCH_SIZE):
            batch_features = clip_features[i:i+config.BATCH_SIZE]
            batch_text = batch_features[:, :768]  # First 768 dimensions (text)
            batch_image = batch_features[:, 768:]  # Last 768 dimensions (image)
            
            # Compute attention features
            attention_batch = compute_attention_features(
                torch.FloatTensor(batch_text).to(config.DEVICE),
                torch.FloatTensor(batch_image).to(config.DEVICE),
                concept_embeddings
            )
            attention_features_list.append(attention_batch.cpu())
        
        attention_features = torch.cat(attention_features_list, dim=0).numpy()
        print(f"Attention features shape: {attention_features.shape}")
        
        # Calculate reconstruction errors
        test_dataset = TensorDataset(torch.FloatTensor(attention_features))
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        reconstruction_errors = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(config.DEVICE)
                errors = model.get_reconstruction_error(inputs)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Calculate statistics
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        min_error = np.min(reconstruction_errors)
        max_error = np.max(reconstruction_errors)
        
        # Determine if this is a benign or malicious dataset
        is_benign = dataset_name in ['sciq', 'truthfulqa', 'okvqa_multilang', 'scienceqa', 'mmvet_test']
        
        results[dataset_name] = {
            'num_samples': len(samples),
            'mean_reconstruction_error': float(mean_error),
            'std_reconstruction_error': float(std_error),
            'min_reconstruction_error': float(min_error),
            'max_reconstruction_error': float(max_error),
            'is_benign': is_benign,
            'reconstruction_errors': reconstruction_errors.tolist()
        }
        
        print(f"  Mean reconstruction error: {mean_error:.6f}")
        print(f"  Std reconstruction error: {std_error:.6f}")
        print(f"  Min reconstruction error: {min_error:.6f}")
        print(f"  Max reconstruction error: {max_error:.6f}")
    
    return results

def compute_auroc_auprc(results):
    """Compute AUROC and AUPRC between benign and malicious datasets"""
    print("\n" + "="*60)
    print("COMPUTING AUROC AND AUPRC")
    print("="*60)
    
    # Separate benign and malicious datasets
    benign_datasets = {k: v for k, v in results.items() if v['is_benign']}
    malicious_datasets = {k: v for k, v in results.items() if not v['is_benign']}
    
    if not benign_datasets or not malicious_datasets:
        print("Error: Need both benign and malicious datasets for AUROC/AUPRC computation")
        return None, None
    
    # Collect all reconstruction errors and labels
    all_errors = []
    all_labels = []  # 0 for benign, 1 for malicious
    
    # Add benign samples
    for dataset_name, data in benign_datasets.items():
        errors = data['reconstruction_errors']
        all_errors.extend(errors)
        all_labels.extend([0] * len(errors))
        print(f"  {dataset_name}: {len(errors)} benign samples")
    
    # Add malicious samples
    for dataset_name, data in malicious_datasets.items():
        errors = data['reconstruction_errors']
        all_errors.extend(errors)
        all_labels.extend([1] * len(errors))
        print(f"  {dataset_name}: {len(errors)} malicious samples")
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    print(f"\nTotal samples: {len(all_errors)}")
    print(f"Benign samples: {np.sum(all_labels == 0)}")
    print(f"Malicious samples: {np.sum(all_labels == 1)}")
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(all_labels, all_errors)
        print(f"\nAUROC: {auroc:.4f}")
    except Exception as e:
        print(f"Error computing AUROC: {e}")
        auroc = None
    
    # Compute AUPRC
    try:
        auprc = average_precision_score(all_labels, all_errors)
        print(f"AUPRC: {auprc:.4f}")
    except Exception as e:
        print(f"Error computing AUPRC: {e}")
        auprc = None
    
    # Compute ROC curve for additional analysis
    try:
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_errors)
        print(f"ROC curve computed with {len(fpr)} points")
    except Exception as e:
        print(f"Error computing ROC curve: {e}")
        fpr, tpr, roc_thresholds = None, None, None
    
    # Compute Precision-Recall curve
    try:
        precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_errors)
        print(f"Precision-Recall curve computed with {len(precision)} points")
    except Exception as e:
        print(f"Error computing Precision-Recall curve: {e}")
        precision, recall, pr_thresholds = None, None, None
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'roc_curve': {
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None,
            'thresholds': roc_thresholds.tolist() if roc_thresholds is not None else None
        },
        'pr_curve': {
            'precision': precision.tolist() if precision is not None else None,
            'recall': recall.tolist() if recall is not None else None,
            'thresholds': pr_thresholds.tolist() if pr_thresholds is not None else None
        },
        'total_samples': len(all_errors),
        'benign_samples': int(np.sum(all_labels == 0)),
        'malicious_samples': int(np.sum(all_labels == 1))
    }

def analyze_distribution_shift(results):
    """Analyze the distribution shift between MM-Vet training and test datasets"""
    print("\n" + "="*60)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*60)
    
    # Separate benign and malicious datasets
    benign_datasets = {k: v for k, v in results.items() if v['is_benign']}
    malicious_datasets = {k: v for k, v in results.items() if not v['is_benign']}
    
    print(f"\nBenign datasets: {list(benign_datasets.keys())}")
    print(f"Malicious datasets: {list(malicious_datasets.keys())}")
    
    # Calculate statistics
    if benign_datasets:
        benign_errors = [v['mean_reconstruction_error'] for v in benign_datasets.values()]
        print(f"\nBenign datasets reconstruction errors:")
        for name, error in zip(benign_datasets.keys(), benign_errors):
            print(f"  {name}: {error:.6f}")
        
        print(f"  Mean across benign: {np.mean(benign_errors):.6f}")
        print(f"  Std across benign: {np.std(benign_errors):.6f}")
        print(f"  Range: {np.min(benign_errors):.6f} - {np.max(benign_errors):.6f}")
    
    if malicious_datasets:
        malicious_errors = [v['mean_reconstruction_error'] for v in malicious_datasets.values()]
        print(f"\nMalicious datasets reconstruction errors:")
        for name, error in zip(malicious_datasets.keys(), malicious_errors):
            print(f"  {name}: {error:.6f}")
        
        print(f"  Mean across malicious: {np.mean(malicious_errors):.6f}")
        print(f"  Std across malicious: {np.std(malicious_errors):.6f}")
        print(f"  Range: {np.min(malicious_errors):.6f} - {np.max(malicious_errors):.6f}")
    
    # Check for potential false positives (benign datasets with high errors)
    if benign_datasets:
        high_error_threshold = np.mean(benign_errors) + 2 * np.std(benign_errors)
        print(f"\nPotential false positives (benign datasets with high errors > {high_error_threshold:.6f}):")
        for name, data in benign_datasets.items():
            if data['mean_reconstruction_error'] > high_error_threshold:
                print(f"  {name}: {data['mean_reconstruction_error']:.6f} (HIGH ERROR)")
            else:
                print(f"  {name}: {data['mean_reconstruction_error']:.6f} (normal)")
    
    # Check for potential false negatives (malicious datasets with low errors)
    if malicious_datasets:
        low_error_threshold = np.mean(malicious_errors) - 2 * np.std(malicious_errors)
        print(f"\nPotential false negatives (malicious datasets with low errors < {low_error_threshold:.6f}):")
        for name, data in malicious_datasets.items():
            if data['mean_reconstruction_error'] < low_error_threshold:
                print(f"  {name}: {data['mean_reconstruction_error']:.6f} (LOW ERROR)")
            else:
                print(f"  {name}: {data['mean_reconstruction_error']:.6f} (normal)")

def main():
    print("="*80)
    print("MM-VET ONLY TRAINING EXPERIMENT")
    print("="*80)
    print("Training: MM-Vet 80% (multimodal benign) with Official JailDAM Algorithm")
    print("Testing: MM-Vet 20%, SciQ, TruthfulQA, OK-VQA-multilang, ScienceQA (benign) + JailBreakV-28k subset (malicious)")
    print("Algorithm: Attention features + Concept embeddings + Small autoencoder")
    print("="*80)
    
    # Set random seed
    config = MMVetOnlyConfig()
    set_seed(config.SEED)
    print(f"Using device: {config.DEVICE}")
    print(f"Random seed set to: {config.SEED}")
    
    # Load MM-Vet data and split
    print("\n" + "="*50)
    print("LOADING MM-VET DATA")
    print("="*50)
    
    mmvet_train_samples, mmvet_test_samples = load_mmvet_data()
    if not mmvet_train_samples:
        print("Error: No MM-Vet training samples loaded!")
        sys.exit(1)
    
    # Load test datasets
    print("\n" + "="*50)
    print("LOADING TEST DATASETS")
    print("="*50)
    
    test_data = load_all_test_datasets(config.MAX_SAMPLES_PER_DATASET)
    
    # Load additional JailBreakV samples
    print("Loading additional JailBreakV samples...")
    jailbreak_samples = load_jailbreakv_subset(config.MAX_JAILBREAK_SAMPLES)
    if jailbreak_samples:
        test_data['malicious_by_dataset']['jailbreakv_subset'] = jailbreak_samples
        test_data['malicious'].extend(jailbreak_samples)
    
    # Add MM-Vet test samples to benign datasets
    test_data['benign_by_dataset']['mmvet_test'] = mmvet_test_samples
    test_data['benign'].extend(mmvet_test_samples)
    
    # Combine all test datasets
    all_test_datasets = {}
    all_test_datasets.update(test_data['benign_by_dataset'])
    all_test_datasets.update(test_data['malicious_by_dataset'])
    
    print(f"Loaded {len(test_data['benign'])} benign test samples")
    print(f"Loaded {len(test_data['malicious'])} malicious test samples")
    
    # Initialize feature extractor
    print("\n" + "="*50)
    print("EXTRACTING VLM FEATURES")
    print("="*50)
    
    feature_extractor = VLMFeatureExtractor(config.VLM_MODEL, config.DEVICE)
    
    # Generate concept embeddings
    print("\n" + "="*50)
    print("GENERATING CONCEPT EMBEDDINGS")
    print("="*50)
    
    concept_embeddings = generate_concept_embeddings(mmvet_train_samples, config)
    
    # Extract MM-Vet training features
    print("\n" + "="*50)
    print("EXTRACTING MM-VET TRAINING FEATURES")
    print("="*50)
    
    print("Extracting MM-Vet training features...")
    mmvet_features = feature_extractor.extract_features(mmvet_train_samples, batch_size=8)
    print(f"MM-Vet training features shape: {mmvet_features.shape}")
    
    # Train autoencoder on MM-Vet only
    print("\n" + "="*50)
    print("TRAINING AUTOENCODER ON MM-VET ONLY")
    print("="*50)
    
    model, train_losses = train_autoencoder_on_mmvet(mmvet_features, concept_embeddings, config)
    
    # Evaluate on test datasets
    print("\n" + "="*50)
    print("EVALUATING ON TEST DATASETS")
    print("="*50)
    
    results = evaluate_on_test_datasets(model, feature_extractor, all_test_datasets, concept_embeddings, config)
    
    # Compute AUROC and AUPRC
    auroc_auprc_results = compute_auroc_auprc(results)
    
    # Analyze distribution shift
    analyze_distribution_shift(results)
    
    # Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    os.makedirs('results', exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'experiment_name': 'mmvet_only_training',
        'config': {
            'latent_dim': config.LATENT_DIM,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'seed': config.SEED,
            'vlm_model': config.VLM_MODEL,
            'max_samples_per_dataset': config.MAX_SAMPLES_PER_DATASET
        },
        'training_data': {
            'dataset': 'mmvet_train',
            'num_samples': len(mmvet_train_samples),
            'features_shape': list(mmvet_features.shape)
        },
        'mmvet_test_data': {
            'dataset': 'mmvet_test',
            'num_samples': len(mmvet_test_samples)
        },
        'test_results': results,
        'auroc_auprc_results': auroc_auprc_results,
        'analysis': {
            'benign_datasets': [k for k, v in results.items() if v['is_benign']],
            'malicious_datasets': [k for k, v in results.items() if not v['is_benign']],
            'benign_mean_errors': [v['mean_reconstruction_error'] for k, v in results.items() if v['is_benign']],
            'malicious_mean_errors': [v['mean_reconstruction_error'] for k, v in results.items() if not v['is_benign']]
        }
    }
    
    with open('results/mmvet_only_experiment_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("Results saved to results/mmvet_only_experiment_results.json")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print("This experiment investigates whether training on MM-Vet only leads to")
    print("rejection of other benign datasets with different distributions.")
    print("Check the results for potential false positives and false negatives.")

if __name__ == "__main__":
    main()
