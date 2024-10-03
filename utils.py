import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.optim as optim

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.imshow(image, cmap='viridis')  # Specify the colormap if needed
        plt.colorbar()
        # Randomly select locations to print values
        x_locs = np.random.randint(0, image.shape[1], 5)
        y_locs = np.random.randint(0, image.shape[0], 5)
        for x, y in zip(x_locs, y_locs):
            plt.text(x, y, f'{image[y, x]:.2f}', color='red')  # Print values at random locations

    plt.show()
# For one pair only.
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
                        
                        #data normalization#
def znorm(data):
    mean, std = np.mean(data), np.std(data)
    return (data - mean) / std, mean, std
def iznorm(data, mean, std):
    return data * std + mean
def lognorm(data):
    min_val = np.min(data)
    return np.log(data - min_val + 0.001), min_val, min_val
def norm(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val
def inorm(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val