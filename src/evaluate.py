# evaluate.py
# This is our REPORT CARD ROBOT! 📊
# It checks how well the AI brains learned and creates beautiful reports
# Like a teacher grading tests and making charts!

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def calculate_metrics(model, test_loader, device):
    """
    This calculates all the important scores for our model!
    Like getting your test score, attendance, and participation grade 📝
    
    Returns: Dictionary with accuracy, F1-score, and predictions
    """
    print("📊 Calculating detailed metrics...")
    
    model.eval()  # Put model in evaluation mode
    
    all_predictions = []
    all_labels = []
    
    # Make predictions on all test data
    with torch.no_grad():
        for text, labels in test_loader:
            text = text.to(device)
            labels = labels.to(device)
            
            # Get predictions
            predictions = model(text).squeeze(1)
            
            # Convert to binary (0 or 1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            
            # Store predictions and true labels
            all_predictions.extend(rounded_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    This creates a confusion matrix picture!
    It shows what the model got right and wrong 🎨
    
    labels: True answers
    predictions: Model's guesses
    save_path: Where to save the picture
    """
    print("🎨 Creating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    This creates pictures showing how the model learned over time!
    Like a growth chart showing your progress 📈
    
    history: Training history dictionary
    save_path: Where to save the picture
    """
    print("🎨 Creating training history plots...")
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss over time
    ax1.plot(epochs, history['train_losses'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_losses'], 'r-o', label='Test Loss', linewidth=2)
    ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over time
    ax2.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 
             'g-o', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved training history to {save_path}")
    
    plt.close()


def plot_comparison_charts(results_df, save_dir='results/plots'):
    """
    This creates comparison charts for all experiments!
    Like making a poster comparing all your science experiments 🔬
    
    results_df: DataFrame with all experimental results
    save_dir: Where to save the charts
    """
    print("🎨 Creating comparison charts...")
    
    # Make sure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Chart 1: Accuracy vs Sequence Length
    if 'sequence_length' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Group by architecture and sequence length
        for arch in results_df['architecture'].unique():
            data = results_df[results_df['architecture'] == arch]
            plt.plot(data['sequence_length'], data['accuracy'] * 100, 
                    marker='o', linewidth=2, label=arch.upper())
        
        plt.title('Accuracy vs Sequence Length', fontsize=16, fontweight='bold')
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, 'accuracy_vs_sequence_length.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved to {save_path}")
        plt.close()
    
    # Chart 2: F1-Score vs Sequence Length
    if 'sequence_length' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        for arch in results_df['architecture'].unique():
            data = results_df[results_df['architecture'] == arch]
            plt.plot(data['sequence_length'], data['f1_score'], 
                    marker='o', linewidth=2, label=arch.upper())
        
        plt.title('F1-Score vs Sequence Length', fontsize=16, fontweight='bold')
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, 'f1_vs_sequence_length.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved to {save_path}")
        plt.close()
    
    # Chart 3: Training Time Comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar chart for training time
    architectures = results_df['architecture'].unique()
    x_pos = np.arange(len(results_df))
    
    plt.bar(x_pos, results_df['avg_epoch_time'], color='skyblue', edgecolor='navy')
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('Average Epoch Time (seconds)', fontsize=12)
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, range(len(results_df)), rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(save_dir, 'training_time_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to {save_path}")
    plt.close()
    
    print("✅ All comparison charts created!")


def print_classification_report(labels, predictions):
    """
    This prints a detailed report of model performance!
    Like a detailed report card with all subjects 📋
    
    labels: True answers
    predictions: Model's guesses
    """
    print("\n" + "="*60)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(labels, predictions, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    print(report)
    
    print("="*60 + "\n")


def save_results_to_csv(results, filename='results/metrics.csv'):
    """
    This saves all results to a CSV file!
    Like keeping a gradebook with all scores 📓
    
    results: Dictionary with results
    filename: Where to save
    """
    print(f"💾 Saving results to {filename}...")
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([results])
    
    # Check if file exists
    file_exists = os.path.isfile(filename)
    
    # Append to CSV
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    print(f"✅ Results saved successfully!")


if __name__ == "__main__":
    """
    This tests the evaluation functions!
    Like practicing grading papers before the real test! 🧪
    """
    print("🧪 Testing evaluation functions...\n")
    
    # Create fake data for testing
    fake_labels = np.random.randint(0, 2, 100)
    fake_predictions = np.random.randint(0, 2, 100)
    
    # Test confusion matrix
    plot_confusion_matrix(fake_labels, fake_predictions, 
                         save_path='results/plots/test_confusion_matrix.png')
    
    # Test training history plot
    fake_history = {
        'train_losses': [0.5, 0.4, 0.3, 0.25, 0.2],
        'test_losses': [0.55, 0.45, 0.35, 0.3, 0.28],
        'test_accuracies': [0.7, 0.75, 0.8, 0.82, 0.85]
    }
    plot_training_history(fake_history, 
                         save_path='results/plots/test_training_history.png')
    
    # Test classification report
    print_classification_report(fake_labels, fake_predictions)
    
    print("✅ All evaluation tests passed!")