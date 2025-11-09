# create_final_plots.py
# FIXED VERSION - Gradient Clipping NaN Issue Resolved

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("📊 Creating final comparison charts...")

# Load the data
df = pd.read_csv('results/metrics.csv')
print(f"📈 Loaded data with {len(df)} experiments")

# Make sure plots directory exists
os.makedirs('../results/plots', exist_ok=True)

# Chart 1: Architecture Comparison (RNN vs LSTM vs BiLSTM)
print("📈 Chart 1: Architecture Comparison...")
arch_data = df[df['architecture'].isin(['rnn', 'lstm', 'bilstm']) & 
               (df['optimizer'] == 'adam') & 
               (df['sequence_length'] == 50)].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy comparison
ax1.bar(arch_data['architecture'].str.upper(), arch_data['accuracy'] * 100, 
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy by Architecture', fontsize=14, fontweight='bold')
ax1.set_ylim(50, 85)
for i, (arch, acc) in enumerate(zip(arch_data['architecture'].str.upper(), arch_data['accuracy'] * 100)):
    ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontweight='bold')

# Training time comparison
ax2.bar(arch_data['architecture'].str.upper(), arch_data['avg_epoch_time'], 
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Avg Epoch Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Training Time by Architecture', fontsize=14, fontweight='bold')
for i, (arch, time) in enumerate(zip(arch_data['architecture'].str.upper(), arch_data['avg_epoch_time'])):
    ax2.text(i, time + 5, f'{time:.0f}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/architecture_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: architecture_comparison.png")
plt.close()

# Chart 2: Optimizer Comparison
print("📈 Chart 2: Optimizer Comparison...")
opt_data = df[df['optimizer'].isin(['adam', 'sgd', 'rmsprop']) & 
              (df['architecture'] == 'lstm') & 
              (df['sequence_length'] == 50)].copy()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ECC71', '#E74C3C', '#F39C12']
bars = ax.bar(opt_data['optimizer'].str.upper(), opt_data['accuracy'] * 100, 
              color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Optimizer', fontsize=14, fontweight='bold')
ax.set_ylim(40, 85)
for i, (opt, acc) in enumerate(zip(opt_data['optimizer'].str.upper(), opt_data['accuracy'] * 100)):
    ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: optimizer_comparison.png")
plt.close()

# Chart 3: Sequence Length Impact
print("📈 Chart 3: Sequence Length Impact...")
seq_data = df[df['sequence_length'].isin([25, 50, 100]) & 
              (df['architecture'] == 'lstm') & 
              (df['optimizer'] == 'adam')].copy()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_data['sequence_length'], seq_data['accuracy'] * 100, 
        marker='o', linewidth=3, markersize=12, color='#9B59B6')
ax.set_xlabel('Sequence Length (words)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Sequence Length on Accuracy', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(60, 85)
for x, y in zip(seq_data['sequence_length'], seq_data['accuracy'] * 100):
    ax.text(x, y + 1, f'{y:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/sequence_length_impact.png', dpi=300, bbox_inches='tight')
print("✅ Saved: sequence_length_impact.png")
plt.close()

# Chart 4: Activation Function Comparison
print("📈 Chart 4: Activation Function Comparison...")
act_data = df[df['activation'].isin(['sigmoid', 'relu', 'tanh']) & 
              (df['architecture'] == 'lstm') & 
              (df['optimizer'] == 'adam') &
              (df['sequence_length'] == 50)].copy()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(act_data['activation'].str.capitalize(), act_data['accuracy'] * 100, 
              color=['#E67E22', '#3498DB', '#1ABC9C'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Activation Function', fontsize=14, fontweight='bold')
ax.set_ylim(70, 80)
for i, (act, acc) in enumerate(zip(act_data['activation'].str.capitalize(), act_data['accuracy'] * 100)):
    ax.text(i, acc + 0.3, f'{acc:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/activation_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: activation_comparison.png")
plt.close()

# Chart 5: Gradient Clipping Comparison - FIXED VERSION
print("📈 Chart 5: Gradient Clipping Comparison...")
# Get the specific experiments for gradient clipping comparison
clip_data = df[(df['architecture'] == 'lstm') & 
               (df['optimizer'] == 'adam') &
               (df['sequence_length'] == 50) &
               (df['activation'] == 'tanh')].copy()

print("Available gradient_clipping values:", clip_data['gradient_clipping'].unique())

# Handle NaN values for "no clipping"
with_clip_acc = clip_data[clip_data['gradient_clipping'] == 1.0]['accuracy'].values[0] * 100
no_clip_acc = clip_data[clip_data['gradient_clipping'].isna()]['accuracy'].values[0] * 100

fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Without Clipping', 'With Clipping']
values = [no_clip_acc, with_clip_acc]
bars = ax.bar(labels, values, color=['#E74C3C', '#2ECC71'], 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Gradient Clipping', fontsize=14, fontweight='bold')
ax.set_ylim(70, 80)
for i, v in enumerate(values):
    ax.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/gradient_clipping_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: gradient_clipping_comparison.png")
plt.close()

# Chart 6: Overall Best Models
print("📈 Chart 6: Top 5 Best Models...")
top_models = df.nlargest(5, 'accuracy')

fig, ax = plt.subplots(figsize=(12, 6))
model_names = [f"{row['architecture'].upper()}\n{row['optimizer'].upper()}\nSeq:{int(row['sequence_length'])}" 
               for _, row in top_models.iterrows()]
bars = ax.barh(model_names, top_models['accuracy'] * 100, 
               color=plt.cm.viridis(range(5)), edgecolor='black', linewidth=1.5)
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 5 Best Performing Models', fontsize=14, fontweight='bold')
ax.set_xlim(70, 85)
for i, (name, acc) in enumerate(zip(model_names, top_models['accuracy'] * 100)):
    ax.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/top_models.png', dpi=300, bbox_inches='tight')
print("✅ Saved: top_models.png")
plt.close()

print("\n🎉 All charts created successfully!")
print("📁 Check results/plots/ folder for all visualizations!")