import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
from torch.utils.data import DataLoader
from data_generator import BrainTumorDataset
from model import UNet

# Set device
# (остальной код без изменений) 