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

#Visualization with common bar and same location values
def visual(Bar=False, **images):
    """Plot images in one row with a single color bar."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    
    # Find the common color range
    vmin = min(image.min() for image in images.values())
    vmax = max(image.max() for image in images.values())
    
    # Generate random locations once
    first_image = list(images.values())[0]
    x_locs = np.random.randint(0, first_image.shape[1], 5)
    y_locs = np.random.randint(0, first_image.shape[0], 5)
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        img = plt.imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)  # Use common color range
        
        # Print values at the same random locations
        #for x, y in zip(x_locs, y_locs):
        #    plt.text(x, y, f'{image[y, x]:.2f}', color='red')  # Print values at random locations
    
    if Bar:
        cbar = plt.colorbar(img, ax=plt.gca(), orientation='horizontal', fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
    
    plt.show()

def visualize(Bar=False, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.imshow(image, cmap='viridis')  # Specify the colormap if needed
        if(Bar==True):
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

#atention loss
def attention_loss(output, target):
    # Calculate MSE loss
    mse_loss = F.mse_loss(output, target)
    # Compute attention weights based on output values
    attention_weights = torch.abs(output)
    # Normalize attention weights to sum to 1
    attention_weights /= attention_weights.sum()
    # Combine MSE loss with attention weights
    attention_aware_loss = torch.sum(attention_weights * mse_loss)
    return attention_aware_loss