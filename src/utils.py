# utils.py
# This is our HELPER ROBOT! 🤖
# It has special tools that other robots will use
# Think of it like a toolbox with hammers and screwdrivers!

import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    This makes sure our experiment gives the same results every time!
    Like making sure a recipe always tastes the same 🍪
    
    seed: A magic number (we use 42) that controls randomness
    """
    # Set the seed for PyTorch (for making neural networks)
    torch.manual_seed(seed)
    
    # Set the seed for NumPy (for math with arrays)
    np.random.seed(seed)
    
    # Set the seed for Python's random module
    random.seed(seed)
    
    # If using GPU (fancy fast computer), set its seed too
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"🌱 Random seed set to {seed} - Results will be the same every time!")


def get_device():
    """
    This checks if we have a super-fast GPU or just a regular CPU
    Like checking if we have a race car or a bicycle 🚗 vs 🚲
    
    Returns: The device to use (cuda for GPU, cpu for CPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("🐢 Using CPU (this will be slower, but it works!)")
    
    return device


def save_results(results, filename):
    """
    This saves our experiment results to a file!
    Like writing down your test scores in a notebook 📝
    
    results: A dictionary with all our results
    filename: Where to save it
    """
    import pandas as pd
    import os
    
    # Make sure the results folder exists
    os.makedirs('results', exist_ok=True)
    
    # Convert results to a table and save
    df = pd.DataFrame([results])
    
    # Check if file exists to add headers or not
    file_exists = os.path.isfile(filename)
    
    # Save to CSV file
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    print(f"💾 Results saved to {filename}")


def format_time(seconds):
    """
    This converts seconds into a nice readable format
    Like saying "2 minutes and 30 seconds" instead of "150 seconds"
    
    seconds: Number of seconds
    Returns: A nice string like "2m 30s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"