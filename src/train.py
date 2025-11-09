# train.py
# This is our TRAINING COACH! 🏋️‍♂️
# It teaches the AI brains to understand if reviews are positive or negative
# Like a teacher helping students learn!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from models import get_model
from preprocess import load_imdb_data, preprocess_data
from utils import set_seed, get_device, format_time

def create_data_loaders(train_data, train_labels, test_data, test_labels, batch_size=32):
    """
    This organizes the data into small batches for training
    Like dividing a big pile of homework into smaller chunks! 📚
    
    batch_size: How many reviews to look at once (32)
    Returns: Data loaders for training and testing
    """
    print(f"📦 Creating data batches of size {batch_size}...")
    
    # Convert numpy arrays to PyTorch tensors (special format for neural networks)
    train_data = torch.LongTensor(train_data)
    train_labels = torch.FloatTensor(train_labels)
    test_data = torch.LongTensor(test_data)
    test_labels = torch.FloatTensor(test_labels)
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Create data loaders (they feed data to the model in batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"✅ Created {len(train_loader)} training batches")
    print(f"✅ Created {len(test_loader)} testing batches")
    
    return train_loader, test_loader


def get_optimizer(model, optimizer_name='adam', learning_rate=0.001):
    """
    This chooses how the model learns from mistakes
    Like choosing whether to learn slowly and carefully or quickly!
    
    optimizer_name: Which learning method ('adam', 'sgd', or 'rmsprop')
    learning_rate: How big each learning step is
    Returns: The optimizer (learning strategy)
    """
    print(f"🎓 Setting up {optimizer_name.upper()} optimizer with learning rate {learning_rate}...")
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def train_epoch(model, train_loader, optimizer, criterion, device, clip_grad=None):
    """
    This trains the model for ONE round (epoch)
    Like practicing a skill for one day! 📅
    
    model: The AI brain
    train_loader: The training data
    optimizer: The learning strategy
    criterion: How to measure mistakes
    device: CPU or GPU
    clip_grad: Whether to limit how much the model changes (for stability)
    
    Returns: Average loss (how many mistakes)
    """
    # Put model in training mode
    model.train()
    
    epoch_loss = 0
    total_batches = len(train_loader)
    
    # Process each batch of reviews
    for batch_idx, (text, labels) in enumerate(train_loader):
        # Move data to device (CPU or GPU)
        text = text.to(device)
        labels = labels.to(device)
        
        # Clear previous gradients (start fresh)
        optimizer.zero_grad()
        
        # Make predictions
        predictions = model(text).squeeze(1)
        
        # Calculate how wrong we were (loss)
        loss = criterion(predictions, labels)
        
        # Learn from mistakes (backpropagation)
        loss.backward()
        
        # Apply gradient clipping if specified (prevents wild changes)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update the model
        optimizer.step()
        
        # Track total loss
        epoch_loss += loss.item()
        
        # Show progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"   Batch {batch_idx + 1}/{total_batches} - Loss: {loss.item():.4f}")
    
    # Return average loss for this epoch
    return epoch_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """
    This tests how well the model learned!
    Like taking a test after studying 📝
    
    Returns: Average loss and accuracy
    """
    # Put model in evaluation mode (no learning, just testing)
    model.eval()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Don't calculate gradients (we're not learning, just testing)
    with torch.no_grad():
        for text, labels in test_loader:
            # Move data to device
            text = text.to(device)
            labels = labels.to(device)
            
            # Make predictions
            predictions = model(text).squeeze(1)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            # Convert predictions to binary (positive/negative)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            
            # Count correct predictions
            correct += (rounded_preds == labels).sum().item()
            total += labels.size(0)
    
    # Calculate accuracy
    accuracy = correct / total
    
    return epoch_loss / len(test_loader), accuracy


def train_model(model, train_loader, test_loader, optimizer, criterion, 
                device, n_epochs=5, clip_grad=None):
    """
    This is the MAIN TRAINING FUNCTION! 🎯
    It trains the model for multiple rounds and tracks progress
    
    n_epochs: How many times to go through all the data (5 rounds)
    
    Returns: Training history and time taken
    """
    print(f"\n{'='*70}")
    print(f"🚀 Starting training for {n_epochs} epochs...")
    print(f"{'='*70}\n")
    
    # Track history
    train_losses = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    
    # Start timing
    total_start_time = time.time()
    
    # Train for each epoch
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        print(f"📚 Epoch {epoch + 1}/{n_epochs}")
        print("-" * 50)
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                device, clip_grad)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Calculate time for this epoch
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Save history
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print results
        print(f"\n   ✅ Train Loss: {train_loss:.4f}")
        print(f"   ✅ Test Loss: {test_loss:.4f}")
        print(f"   ✅ Test Accuracy: {test_acc*100:.2f}%")
        print(f"   ⏱️  Time: {format_time(epoch_time)}")
        print()
    
    # Calculate total time
    total_time = time.time() - total_start_time
    avg_epoch_time = np.mean(epoch_times)
    
    print(f"{'='*70}")
    print(f"🎉 Training Complete!")
    print(f"⏱️  Total Time: {format_time(total_time)}")
    print(f"⏱️  Average Time per Epoch: {format_time(avg_epoch_time)}")
    print(f"🎯 Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")
    print(f"{'='*70}\n")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'epoch_times': epoch_times,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'final_accuracy': test_accuracies[-1]
    }


if __name__ == "__main__":
    """
    This is a quick test to make sure training works!
    Like a practice run before the real race! 🏁
    """
    print("🧪 Testing training pipeline...\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load and preprocess data
    print("\n📚 Loading data...")
    train_data, train_labels, test_data, test_labels = load_imdb_data()
    train_data, test_data = preprocess_data(train_data, test_data, sequence_length=50)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_data, train_labels, test_data, test_labels, batch_size=32
    )
    
    # Create a simple LSTM model for testing
    print("\n🤖 Creating model...")
    model = get_model(model_type='lstm', activation='tanh')
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = get_optimizer(model, optimizer_name='adam', learning_rate=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train for just 2 epochs (quick test)
    print("\n🏋️‍♂️ Training for 2 epochs (quick test)...")
    history = train_model(model, train_loader, test_loader, optimizer, 
                         criterion, device, n_epochs=2, clip_grad=1.0)
    
    print("✅ Training test complete!")