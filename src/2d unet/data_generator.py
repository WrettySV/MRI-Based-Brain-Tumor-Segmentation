import os
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

# Constants
MODALITIES = {
    't1': 'T1-weighted - shows normal anatomy',
    't1ce': 'T1-weighted with Contrast Enhancement - highlights active tumor regions',
    't2': 'T2-weighted - shows edema and tumor regions',
    'flair': 'Fluid Attenuated Inversion Recovery - highlights edema and tumor regions'
}

SEGMENT_CLASSES = {
    0: 'Background',
    1: 'NCR/NET',
    2: 'ED',
    3: 'ET'
}

VOLUME_SLICES = 100
VOLUME_START_AT = 22

TRAIN_PATH = '../../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
VAL_PATH = '../../BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

def get_train_val_test_split(train_end=300, val_end=350, seed=42):
    """
    Split data into train, validation and test sets
    
    
    Returns:
        train_ids: List of case IDs for training
        val_ids: List of case IDs for validation
        test_ids: List of case IDs for testing
    """
    # Set random seed
    random.seed(seed)
    
    # Get all case IDs from training data
    all_cases = []
    for case_id in os.listdir(TRAIN_PATH):
        if case_id.startswith('BraTS20_Training_'):
            case_path = os.path.join(TRAIN_PATH, case_id)
            if all(os.path.exists(os.path.join(case_path, f'{case_id}_{mod}.nii')) 
                  for mod in ['t1', 't1ce', 't2', 'flair', 'seg']):
                all_cases.append(case_id)
    
    # Sort cases to ensure consistent splitting
    all_cases = sorted(all_cases)
    
    # Split into train, validation and test
    train_ids = [case for case in all_cases if int(case.split('_')[-1]) <= train_end]
    val_ids = [case for case in all_cases if train_end < int(case.split('_')[-1]) <= val_end]
    test_ids = [case for case in all_cases if int(case.split('_')[-1]) > val_end]
    
    print(f"Total cases: {len(all_cases)}")
    print(f"Training cases: {len(train_ids)} (1-{train_end})")
    print(f"Validation cases: {len(val_ids)} ({train_end+1}-{val_end})")
    print(f"Test cases: {len(test_ids)} ({val_end+1}-369)")
    
    return train_ids, val_ids, test_ids

#dim=(128, 128)
class BrainTumorDataset(Dataset):
    def __init__(self, list_IDs, dim=(128, 128), n_channels=4, slices_per_volume=16, is_validation=False):
        """
        PyTorch Dataset for brain tumor segmentation
        
        Args:
            list_IDs: List of case IDs
            dim: Image dimensions (height, width)
            n_channels: Number of input channels (4 for BraTS: T1, T1ce, T2, FLAIR)
            slices_per_volume: Number of slices to take from each volume
            is_validation: Whether this is validation dataset
        """
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.slices_per_volume = slices_per_volume
        self.volume_slices = VOLUME_SLICES
        self.is_validation = is_validation
        
        # Choose correct path
        self.base_path = TRAIN_PATH  # Always use training path, split is handled by list_IDs
        
        # Validate case IDs
        self.valid_cases = []
        for case_id in list_IDs:
            case_path = os.path.join(self.base_path, case_id)
            if all(os.path.exists(os.path.join(case_path, f'{case_id}_{mod}.nii')) 
                  for mod in ['t1', 't1ce', 't2', 'flair', 'seg']):
                self.valid_cases.append(case_id)
        
        if not self.valid_cases:
            raise ValueError(f"No valid cases found in {self.base_path}!")
        
        print(f"Found {len(self.valid_cases)} valid cases out of {len(list_IDs)} in {self.base_path}")
        
        # Calculate total number of samples
        self.total_samples = len(self.valid_cases) * self.slices_per_volume
        
        # Create mapping from index to (volume_idx, slice_idx)
        self.index_to_volume_slice = []
        for i in range(self.total_samples):
            volume_idx = i // self.slices_per_volume
            slice_idx = i % self.slices_per_volume
            self.index_to_volume_slice.append((volume_idx, slice_idx))

    def __len__(self):
        """Total number of samples"""
        return self.total_samples

    def __getitem__(self, index):
        """Generate one sample of data"""
        # Get volume and slice indices
        volume_idx, slice_idx = self.index_to_volume_slice[index]
        case_id = self.valid_cases[volume_idx]
        case_path = os.path.join(self.base_path, case_id)
        
        # Load images
        t1 = nib.load(os.path.join(case_path, f'{case_id}_t1.nii')).get_fdata()
        t1ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
        t2 = nib.load(os.path.join(case_path, f'{case_id}_t2.nii')).get_fdata()
        flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
        seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()


        # Calculate middle slice and range
        middle_slice = self.volume_slices // 2
        half_range = self.slices_per_volume // 2
        
        # Get the specific slice from the middle of the volume
        actual_slice_idx = middle_slice - half_range + slice_idx
        
        # Ensure we don't go out of bounds
        actual_slice_idx = max(0, min(actual_slice_idx, self.volume_slices - 1))
        
        # Prepare input tensor
        X = np.zeros((*self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros(self.dim, dtype=np.float32)

        # Resize and normalize the slice
        X[:, :, 0] = cv2.resize(t1[:, :, actual_slice_idx], self.dim)
        X[:, :, 1] = cv2.resize(t1ce[:, :, actual_slice_idx], self.dim)
        X[:, :, 2] = cv2.resize(t2[:, :, actual_slice_idx], self.dim)
        X[:, :, 3] = cv2.resize(flair[:, :, actual_slice_idx], self.dim)
        y = cv2.resize(seg[:, :, actual_slice_idx], self.dim, interpolation=cv2.INTER_NEAREST)

        # Normalize segmentation labels to [0, 3] range
        y = np.clip(y, 0, 3)

        # Normalize input (avoid division by zero)
        for c in range(self.n_channels):
            max_val = np.max(X[:, :, c])
            if max_val > 0:
                X[:, :, c] = X[:, :, c] / max_val
            else:
                X[:, :, c] = np.zeros_like(X[:, :, c])  # If all zeros, keep as zeros
        
        # Convert to PyTorch tensors
        X = torch.from_numpy(X).permute(2, 0, 1)  # (channels, height, width)
        y = torch.from_numpy(y).long()  # (height, width)
        
        # One-hot encode the labels
        y_one_hot = F.one_hot(y, num_classes=4).permute(2, 0, 1)  # (classes, height, width)
        
        return X, y_one_hot