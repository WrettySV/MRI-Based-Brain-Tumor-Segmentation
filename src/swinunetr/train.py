import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from model import UNet, dice_coefficient, categorical_dice_loss
from data_generator import BrainTumorDataset
import gc

# Constants
TRAIN_PATH = '../../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
BATCH_SIZE = 2  # Keep small batch size
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.2  # 20% for validation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 