# run_experiments.py
# This is the MASTER CONTROLLER! 🎮
# It runs ALL experiments automatically and saves everything!
# Like a science fair coordinator running all your experiments!

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from preprocess import load_imdb_data, preprocess_data
from train import create_data_loaders, get_optimizer, train_model
from evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history, save_results_to_csv
from utils import set_seed, get_device, format_time
import time

def run_single_experiment(config):
    """
    This runs ONE experiment with specific settings!
    Like doing one science experiment at a time 🔬
    
    config: A dictionary with all the settings for this experiment
    Returns: Results dictionary
    """
    print("\n" + "🌟" * 35)
    print(f"🚀 STARTING NEW EXPERIMENT")
    print("🌟" * 35)
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("🌟" * 35 + "\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n📚 Loading and preprocessing data...")
    train_data, train_labels, test_data, test_labels = load_imdb_data()
    train_data, test_data = preprocess_data(
        train_data, test_data, 
        sequence_length=config['sequence_length']
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_data, train_labels, 
        test_data, test_labels, 
        batch_size=config['batch_size']
    )
    
    # Create model
    print(f"\n🤖 Creating {config['architecture'].upper()} model...")
    model = get_model(
        model_type=config['architecture'],
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        activation=config['activation']
    )
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = get_optimizer(
        model, 
        optimizer_name=config['optimizer'],
        learning_rate=config['learning_rate']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Train model
    print("\n🏋️‍♂️ Starting training...")
    start_time = time.time()
    
    history = train_model(
        model, train_loader, test_loader,
        optimizer, criterion, device,
        n_epochs=config['n_epochs'],
        clip_grad=config['gradient_clipping']
    )
    
    total_training_time = time.time() - start_time
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    metrics = calculate_metrics(model, test_loader, device)
    
    # Create unique name for this experiment
    exp_name = f"{config['architecture']}_{config['activation']}_{config['optimizer']}_seq{config['sequence_length']}_clip{config['gradient_clipping']}"
    
    # Save plots
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save training history plot
    plot_training_history(
        history,
        save_path=f"{plots_dir}/{exp_name}_history.png"
    )
    
    # Save confusion matrix
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        save_path=f"{plots_dir}/{exp_name}_confusion.png"
    )
    
    # Compile results
    results = {
        'architecture': config['architecture'],
        'activation': config['activation'],
        'optimizer': config['optimizer'],
        'sequence_length': config['sequence_length'],
        'gradient_clipping': config['gradient_clipping'],
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'avg_epoch_time': history['avg_epoch_time'],
        'total_training_time': total_training_time,
        'final_train_loss': history['train_losses'][-1],
        'final_test_loss': history['test_losses'][-1]
    }
    
    # Save results to CSV
    save_results_to_csv(results, 'results/metrics.csv')
    
    print("\n✅ Experiment completed successfully!")
    print(f"🎯 Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"🎯 F1-Score: {metrics['f1_score']:.4f}")
    print(f"⏱️  Total Time: {format_time(total_training_time)}")
    
    return results


def run_all_experiments():
    """
    This runs ALL the experiments required for the homework!
    Like running your entire science fair! 🎪
    
    We'll test different combinations of:
    - Architectures: RNN, LSTM, BiLSTM
    - Activations: Sigmoid, ReLU, Tanh
    - Optimizers: Adam, SGD, RMSprop
    - Sequence Lengths: 25, 50, 100
    - Gradient Clipping: Yes or No
    """
    
    print("\n" + "🎪" * 35)
    print("🎉 STARTING ALL EXPERIMENTS!")
    print("🎪" * 35 + "\n")
    
    # Base configuration (default settings)
    base_config = {
        'vocab_size': 10000,
        'embedding_dim': 100,
        'hidden_dim': 64,
        'output_dim': 1,
        'n_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'n_epochs': 5,
        'learning_rate': 0.001
    }
    
    # List to store all results
    all_results = []
    experiment_number = 1
    
    # Experiment Set 1: Test different architectures
    print("\n" + "="*70)
    print("📊 EXPERIMENT SET 1: Testing Different Architectures")
    print("="*70)
    
    for architecture in ['rnn', 'lstm', 'bilstm']:
        config = base_config.copy()
        config['architecture'] = architecture
        config['activation'] = 'tanh'
        config['optimizer'] = 'adam'
        config['sequence_length'] = 50
        config['gradient_clipping'] = 1.0
        
        print(f"\n🔬 Experiment {experiment_number}: {architecture.upper()}")
        results = run_single_experiment(config)
        all_results.append(results)
        experiment_number += 1
    
    # Experiment Set 2: Test different activations (using best architecture)
    print("\n" + "="*70)
    print("📊 EXPERIMENT SET 2: Testing Different Activation Functions")
    print("="*70)
    
    for activation in ['sigmoid', 'relu', 'tanh']:
        config = base_config.copy()
        config['architecture'] = 'lstm'  # Using LSTM as it's usually best
        config['activation'] = activation
        config['optimizer'] = 'adam'
        config['sequence_length'] = 50
        config['gradient_clipping'] = 1.0
        
        print(f"\n🔬 Experiment {experiment_number}: {activation.upper()} activation")
        results = run_single_experiment(config)
        all_results.append(results)
        experiment_number += 1
    
    # Experiment Set 3: Test different optimizers
    print("\n" + "="*70)
    print("📊 EXPERIMENT SET 3: Testing Different Optimizers")
    print("="*70)
    
    for optimizer in ['adam', 'sgd', 'rmsprop']:
        config = base_config.copy()
        config['architecture'] = 'lstm'
        config['activation'] = 'tanh'
        config['optimizer'] = optimizer
        config['sequence_length'] = 50
        config['gradient_clipping'] = 1.0
        
        print(f"\n🔬 Experiment {experiment_number}: {optimizer.upper()} optimizer")
        results = run_single_experiment(config)
        all_results.append(results)
        experiment_number += 1
    
    # Experiment Set 4: Test different sequence lengths
    print("\n" + "="*70)
    print("📊 EXPERIMENT SET 4: Testing Different Sequence Lengths")
    print("="*70)
    
    for seq_length in [25, 50, 100]:
        config = base_config.copy()
        config['architecture'] = 'lstm'
        config['activation'] = 'tanh'
        config['optimizer'] = 'adam'
        config['sequence_length'] = seq_length
        config['gradient_clipping'] = 1.0
        
        print(f"\n🔬 Experiment {experiment_number}: Sequence length {seq_length}")
        results = run_single_experiment(config)
        all_results.append(results)
        experiment_number += 1
    
    # Experiment Set 5: Test gradient clipping
    print("\n" + "="*70)
    print("📊 EXPERIMENT SET 5: Testing Gradient Clipping")
    print("="*70)
    
    for clip_value in [None, 1.0]:
        config = base_config.copy()
        config['architecture'] = 'lstm'
        config['activation'] = 'tanh'
        config['optimizer'] = 'adam'
        config['sequence_length'] = 50
        config['gradient_clipping'] = clip_value
        
        clip_name = "No Clipping" if clip_value is None else f"Clipping={clip_value}"
        print(f"\n🔬 Experiment {experiment_number}: {clip_name}")
        results = run_single_experiment(config)
        all_results.append(results)
        experiment_number += 1
    
    # Print final summary
    print("\n" + "🎊" * 35)
    print("🎉 ALL EXPERIMENTS COMPLETED!")
    print("🎊" * 35)
    print(f"\n✅ Total experiments run: {len(all_results)}")
    print(f"📁 Results saved to: results/metrics.csv")
    print(f"📊 Plots saved to: results/plots/")
    print("\n🌟 AMAZING JOB! YOU DID IT! 🌟\n")


if __name__ == "__main__":
    """
    This is where everything starts!
    The main entry point! 🚪
    """
    print("\n" + "⭐" * 35)
    print("🎓 SENTIMENT ANALYSIS PROJECT")
    print("🎬 IMDb Movie Review Classification")
    print("⭐" * 35 + "\n")
    
    # Run all experiments
    run_all_experiments()