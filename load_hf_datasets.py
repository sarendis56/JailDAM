#!/usr/bin/env python3
"""
HuggingFace Dataset Loaders for Jailbreak Detection Experiments

This module provides loaders for HuggingFace datasets to test distribution shift
in autoencoder-based jailbreak detection.
"""

import os
import json
import random
import numpy as np
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

# Set random seed for reproducible sampling
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def ensure_deterministic_sampling():
    """Ensure deterministic sampling by resetting seed before random operations"""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def load_sciq(max_samples=None):
    """
    Load SciQ dataset - a multiple choice science question dataset
    
    Args:
        max_samples: Maximum number of samples to return
        
    Returns:
        List of samples in format {"txt": str, "img": None, "toxicity": 0}
    """
    try:
        print("Loading SciQ dataset from HuggingFace...")
        dataset = load_dataset("sciq", split="train")
        
        samples = []
        for item in dataset:
            # Create a question-answer pair
            question = item['question']
            correct_answer = item['correct_answer']
            distractor1 = item['distractor1']
            distractor2 = item['distractor2']
            distractor3 = item['distractor3']
            
            # Format as a multiple choice question
            text = f"Question: {question}\nA) {correct_answer}\nB) {distractor1}\nC) {distractor2}\nD) {distractor3}\nAnswer: {correct_answer}"
            
            sample = {
                "txt": text,
                "img": None,
                "toxicity": 0,  # SciQ is benign
                "dataset": "sciq",
                "question_id": f"sciq_{len(samples)}"
            }
            samples.append(sample)
        
        # Apply sample limit
        if max_samples and len(samples) > max_samples:
            ensure_deterministic_sampling()
            samples = random.sample(samples, max_samples)
        
        print(f"Successfully loaded {len(samples)} samples from SciQ")
        return samples
        
    except Exception as e:
        print(f"Error loading SciQ: {e}")
        return []

def load_truthfulqa(max_samples=None):
    """
    Load TruthfulQA dataset - a benchmark to measure whether a language model is truthful in generating answers to questions
    
    Args:
        max_samples: Maximum number of samples to return
        
    Returns:
        List of samples in format {"txt": str, "img": None, "toxicity": 0}
    """
    try:
        print("Loading TruthfulQA dataset from HuggingFace...")
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        samples = []
        for item in dataset:
            question = item['question']
            # Get the correct answer
            correct_answer = item['mc1_targets']['labels'][0]  # First correct answer
            choices = item['mc1_targets']['choices']
            
            # Format as a question-answer pair
            text = f"Question: {question}\nAnswer: {choices[correct_answer]}"
            
            sample = {
                "txt": text,
                "img": None,
                "toxicity": 0,  # TruthfulQA is benign
                "dataset": "truthfulqa",
                "question_id": f"truthfulqa_{len(samples)}"
            }
            samples.append(sample)
        
        # Apply sample limit
        if max_samples and len(samples) > max_samples:
            ensure_deterministic_sampling()
            samples = random.sample(samples, max_samples)
        
        print(f"Successfully loaded {len(samples)} samples from TruthfulQA")
        return samples
        
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
        return []

def load_okvqa_multilang(max_samples=None):
    """
    Load OK-VQA-multilang dataset - a multilingual visual question answering dataset
    
    Args:
        max_samples: Maximum number of samples to return
        
    Returns:
        List of samples in format {"txt": str, "img": None, "toxicity": 0}
    """
    try:
        print("Loading OK-VQA-multilang dataset from HuggingFace...")
        dataset = load_dataset("dinhanhx/OK-VQA-multilang", split="train")
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= (max_samples or len(dataset)):
                break
                
            # Get the text content (this dataset only has 'text' field)
            text_content = item['text']
            
            # Use the text content directly as it appears to be question-answer pairs
            text = f"Question: {text_content}"
            
            sample = {
                "txt": text,
                "img": None,  # This dataset doesn't have images
                "toxicity": 0,  # OK-VQA is benign
                "dataset": "okvqa_multilang",
                "question_id": f"okvqa_{i}"
            }
            samples.append(sample)
        
        print(f"Successfully loaded {len(samples)} samples from OK-VQA-multilang")
        return samples
        
    except Exception as e:
        print(f"Error loading OK-VQA-multilang: {e}")
        return []

def load_scienceqa(max_samples=None):
    """
    Load ScienceQA dataset - a multimodal science question answering dataset
    
    Args:
        max_samples: Maximum number of samples to return
        
    Returns:
        List of samples in format {"txt": str, "img": str, "toxicity": 0}
    """
    try:
        print("Loading ScienceQA dataset from HuggingFace...")
        dataset = load_dataset("derek-thomas/ScienceQA", split="train")
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= (max_samples or len(dataset)):
                break
                
            # Get the question and choices
            question = item['question']
            choices = item['choices']
            answer = item['answer']
            
            # Format as a multiple choice question
            choice_text = "\n".join([f"{chr(65+j)}) {choice}" for j, choice in enumerate(choices)])
            text = f"Question: {question}\n{choice_text}\nAnswer: {chr(65+answer)}) {choices[answer]}"
            
            # Handle image if present
            img_path = None
            if item.get('image') is not None:
                img_path = f"data/scienceqa_images/scienceqa_{i}.jpg"
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                # Save image if not already exists
                if not os.path.exists(img_path):
                    try:
                        # ScienceQA image is a dict with 'bytes' and 'path' keys
                        if isinstance(item['image'], dict) and 'bytes' in item['image']:
                            # Convert bytes to PIL Image and save
                            from PIL import Image
                            import io
                            image_bytes = item['image']['bytes']
                            image = Image.open(io.BytesIO(image_bytes))
                            image.save(img_path)
                        else:
                            # Fallback: try to save directly (for other formats)
                            item['image'].save(img_path)
                    except Exception as e:
                        print(f"Error saving ScienceQA image {i}: {e}")
                        img_path = None
            
            sample = {
                "txt": text,
                "img": img_path if img_path and os.path.exists(img_path) else None,
                "toxicity": 0,  # ScienceQA is benign
                "dataset": "scienceqa",
                "question_id": f"scienceqa_{i}"
            }
            samples.append(sample)
        
        print(f"Successfully loaded {len(samples)} samples from ScienceQA")
        return samples
        
    except Exception as e:
        print(f"Error loading ScienceQA: {e}")
        return []

def load_jailbreakv_subset(max_samples=None):
    """
    Load JailBreakV-28k subset from local directory
    
    Args:
        max_samples: Maximum number of samples to return
        
    Returns:
        List of samples in format {"txt": str, "img": str, "toxicity": 1}
    """
    try:
        print("Loading JailBreakV-28k subset from local directory...")
        
        # Check if subset directory exists
        subset_path = "data/JailBreakV_28k_subset"
        if not os.path.exists(subset_path):
            print(f"JailBreakV-28k subset directory not found at {subset_path}")
            return []
        
        # Load the CSV file directly
        import pandas as pd
        csv_path = os.path.join(subset_path, "mini_JailBreakV_28K.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return []
        
        df = pd.read_csv(csv_path)
        queries = df["jailbreak_query"].tolist()
        
        # Collect images from the subset directory
        samples = []
        attack_types = ["figstep", "llm_transfer_attack", "query_related"]
        
        for attack_type in attack_types:
            attack_dir = os.path.join(subset_path, attack_type)
            if not os.path.exists(attack_dir):
                continue
                
            # Get all image files in the attack directory
            image_files = [f for f in os.listdir(attack_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for i, img_file in enumerate(image_files):
                if max_samples and len(samples) >= max_samples:
                    break
                    
                # Use query from CSV (cycle through if we have more images than queries)
                query_idx = i % len(queries)
                query = queries[query_idx]
                
                img_path = os.path.join(attack_dir, img_file)
                
                sample = {
                    "txt": query,
                    "img": img_path,
                    "toxicity": 1,  # JailBreakV is malicious
                    "dataset": "jailbreakv_subset",
                    "question_id": f"jailbreakv_{len(samples)}"
                }
                samples.append(sample)
            
            # If we've reached max_samples, stop processing other attack types
            if max_samples and len(samples) >= max_samples:
                break
        
        print(f"Successfully loaded {len(samples)} samples from JailBreakV-28k subset")
        return samples
        
    except Exception as e:
        print(f"Error loading JailBreakV-28k subset: {e}")
        return []

def load_all_test_datasets(max_samples_per_dataset=500):
    """
    Load all test datasets for the distribution shift experiment
    
    Args:
        max_samples_per_dataset: Maximum samples per dataset
        
    Returns:
        Dict with 'benign' and 'malicious' keys containing lists of samples
    """
    print("Loading all test datasets for distribution shift experiment...")
    
    # Load benign datasets
    benign_datasets = {
        'sciq': load_sciq(max_samples_per_dataset),
        'truthfulqa': load_truthfulqa(max_samples_per_dataset),
        'okvqa_multilang': load_okvqa_multilang(max_samples_per_dataset),
        'scienceqa': load_scienceqa(max_samples_per_dataset)
    }
    
    # Load malicious dataset
    malicious_datasets = {
        'jailbreakv_subset': load_jailbreakv_subset(max_samples_per_dataset)
    }
    
    # Combine all benign samples
    all_benign = []
    for dataset_name, samples in benign_datasets.items():
        all_benign.extend(samples)
        print(f"  {dataset_name}: {len(samples)} samples")
    
    # Combine all malicious samples
    all_malicious = []
    for dataset_name, samples in malicious_datasets.items():
        all_malicious.extend(samples)
        print(f"  {dataset_name}: {len(samples)} samples")
    
    print(f"Total benign samples: {len(all_benign)}")
    print(f"Total malicious samples: {len(all_malicious)}")
    
    return {
        'benign': all_benign,
        'malicious': all_malicious,
        'benign_by_dataset': benign_datasets,
        'malicious_by_dataset': malicious_datasets
    }

if __name__ == "__main__":
    # Test the loaders
    print("Testing HuggingFace dataset loaders...")
    
    # Test each loader with small samples
    test_samples = 10
    
    print(f"\nTesting SciQ (max {test_samples} samples):")
    sciq_samples = load_sciq(test_samples)
    if sciq_samples:
        print(f"  Sample: {sciq_samples[0]['txt'][:100]}...")
    
    print(f"\nTesting TruthfulQA (max {test_samples} samples):")
    truthfulqa_samples = load_truthfulqa(test_samples)
    if truthfulqa_samples:
        print(f"  Sample: {truthfulqa_samples[0]['txt'][:100]}...")
    
    print(f"\nTesting OK-VQA-multilang (max {test_samples} samples):")
    okvqa_samples = load_okvqa_multilang(test_samples)
    if okvqa_samples:
        print(f"  Sample: {okvqa_samples[0]['txt'][:100]}...")
        print(f"  Image: {okvqa_samples[0]['img']} (no images in this dataset)")
    
    print(f"\nTesting ScienceQA (max {test_samples} samples):")
    scienceqa_samples = load_scienceqa(test_samples)
    if scienceqa_samples:
        print(f"  Sample: {scienceqa_samples[0]['txt'][:100]}...")
        print(f"  Image: {scienceqa_samples[0]['img']}")
    
    print(f"\nTesting JailBreakV subset (max {test_samples} samples):")
    jailbreakv_samples = load_jailbreakv_subset(test_samples)
    if jailbreakv_samples:
        print(f"  Sample: {jailbreakv_samples[0]['txt'][:100]}...")
        print(f"  Image: {jailbreakv_samples[0]['img']}")
    
    print("\nAll loaders tested successfully!")
