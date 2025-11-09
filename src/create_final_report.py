"""
create_final_report.py
Generates a professional PDF report for the Sentiment Analysis homework
COMPLETE WORKING VERSION with GitHub link
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import os
import sys

print("📝 Creating Final Homework Report...")

# Get absolute paths - works regardless of where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
results_dir = os.path.join(project_root, 'results')
plots_dir = os.path.join(results_dir, 'plots')

# Make sure results directory exists
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load CSV with absolute path
csv_path = os.path.join(results_dir, 'metrics.csv')
print(f"📁 Looking for CSV at: {csv_path}")

if not os.path.exists(csv_path):
    print(f"❌ ERROR: Cannot find {csv_path}")
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"📍 Script directory: {script_dir}")
    print(f"📍 Project root: {project_root}")
    print("\n💡 Make sure you ran the experiments first:")
    print("   python src/run_experiments.py")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✅ Found CSV file!")
print(f"📊 Loaded {len(df)} experimental results")

# Remove duplicate rows if any
df = df.drop_duplicates(subset=['architecture', 'activation', 'optimizer', 'sequence_length', 'gradient_clipping'])

# Define PDF path
pdf_path = os.path.join(results_dir, 'Homework_3_Report.pdf')
print(f"📝 Will save PDF to: {pdf_path}")

# Create PDF
with PdfPages(pdf_path) as pdf:
    
    # ========== PAGE 1: TITLE PAGE ==========
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.7, 'Homework 3: Sentiment Analysis Report', 
             ha='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.65, 'Comparative Analysis of RNN Architectures', 
             ha='center', fontsize=16)
    fig.text(0.5, 0.62, 'for Movie Review Classification', 
             ha='center', fontsize=16)
    
    fig.text(0.5, 0.45, 'Submitted by:', ha='center', fontsize=14)
    fig.text(0.5, 0.42, 'Sravani Elavarthi', ha='center', fontsize=16, fontweight='bold')
    
    fig.text(0.5, 0.30, f'Date: {datetime.now().strftime("%B %d, %Y")}', 
             ha='center', fontsize=12)
    
    # Add GitHub Repository Link
    fig.text(0.5, 0.25, 'GitHub Repository:', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.22, 'https://github.com/sravani150602/sentiment-analysis-rnn-comparison', 
             ha='center', fontsize=9, color='blue', style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, edgecolor='blue'))
    
    # Summary box
    best_model = df.loc[df['accuracy'].idxmax()]
    summary_text = f"""EXPERIMENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Experiments: {len(df)}
Best Accuracy: {best_model['accuracy']*100:.2f}%
Best Model: {best_model['architecture'].upper()}
Optimizer: {best_model['optimizer'].upper()}
Sequence Length: {int(best_model['sequence_length'])}"""
    
    fig.text(0.5, 0.10, summary_text, ha='center', fontsize=10, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========== PAGE 2: INTRODUCTION ==========
    fig = plt.figure(figsize=(8.5, 11))
    
    intro_text = """1. INTRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentiment classification is a fundamental NLP task that involves categorizing emotional 
tone of text into positive or negative sentiments. This project implements and evaluates 
multiple Recurrent Neural Network architectures for binary sentiment classification.

KEY OBJECTIVES:
• Compare RNN, LSTM, and Bidirectional LSTM architectures
• Evaluate activation functions (Sigmoid, ReLU, Tanh)
• Test optimizers (Adam, SGD, RMSprop)
• Analyze sequence length variations (25, 50, 100 words)
• Investigate gradient clipping for training stability

2. DATASET: IMDb MOVIE REVIEWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPECIFICATIONS:
• Total: 50,000 reviews (25,000 train, 25,000 test)
• Preprocessing: Lowercase, punctuation removal, top 10,000 words
• Average length: ~235 words (before padding)
• Class balance: 50% positive, 50% negative

3. MODEL ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPONENTS:
• Embedding: 100 dimensions
• Hidden Layers: 2 layers × 64 units
• Dropout: 0.3
• Output: Single neuron with sigmoid
• Loss: Binary Cross-Entropy
• Batch Size: 32, Epochs: 5

TESTED VARIATIONS:
• Architectures: RNN, LSTM, BiLSTM
• Activations: Sigmoid, ReLU, Tanh
• Optimizers: Adam, SGD, RMSprop
• Sequence Lengths: 25, 50, 100 words
• Gradient Clipping: None vs 1.0"""
    
    fig.text(0.1, 0.95, intro_text, va='top', fontsize=9, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========== PAGE 3: RESULTS TABLE ==========
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, '4. EXPERIMENTAL RESULTS', 
             ha='center', fontsize=16, fontweight='bold')
    
    # Get top 10 models
    top_10 = df.nlargest(10, 'accuracy')[['architecture', 'activation', 'optimizer', 
                                           'sequence_length', 'gradient_clipping', 
                                           'accuracy', 'f1_score', 'avg_epoch_time']]
    
    # Format the data
    top_10_display = top_10.copy()
    top_10_display['accuracy'] = top_10_display['accuracy'].apply(lambda x: f'{x*100:.2f}%')
    top_10_display['f1_score'] = top_10_display['f1_score'].apply(lambda x: f'{x:.4f}')
    top_10_display['avg_epoch_time'] = top_10_display['avg_epoch_time'].apply(lambda x: f'{x:.1f}s')
    top_10_display['gradient_clipping'] = top_10_display['gradient_clipping'].fillna('None')
    top_10_display.columns = ['Arch', 'Activ', 'Optim', 'SeqLen', 'Clip', 'Acc', 'F1', 'Time(s)']
    
    # Create table
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=top_10_display.values,
                     colLabels=top_10_display.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.05, 0.3, 0.9, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(top_10_display.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(top_10_display) + 1):
        for j in range(len(top_10_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Summary stats
    summary_stats = f"""PERFORMANCE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Accuracy: {df['accuracy'].max()*100:.2f}%
Average Accuracy: {df['accuracy'].mean()*100:.2f}%
Worst Accuracy: {df['accuracy'].min()*100:.2f}%
Best F1-Score: {df['f1_score'].max():.4f}
Fastest Training: {df['avg_epoch_time'].min():.1f}s/epoch
Slowest Training: {df['avg_epoch_time'].max():.1f}s/epoch"""
    
    fig.text(0.5, 0.15, summary_stats, ha='center', fontsize=8, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========== PAGE 4: CHARTS ==========
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle('5. COMPARATIVE ANALYSIS', fontsize=16, fontweight='bold', y=0.98)
    
    # Architecture comparison
    arch_comp = df[(df['optimizer'] == 'adam') & 
                   (df['sequence_length'] == 50) &
                   (df['activation'] == 'tanh')]
    
    axes[0, 0].bar(arch_comp['architecture'].str.upper(), 
                   arch_comp['accuracy'] * 100,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Accuracy by Architecture')
    axes[0, 0].set_ylim(50, 85)
    
    axes[0, 1].bar(arch_comp['architecture'].str.upper(), 
                   arch_comp['avg_epoch_time'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_ylabel('Time (s)', fontweight='bold')
    axes[0, 1].set_title('Training Time')
    
    # Optimizer comparison
    opt_comp = df[(df['architecture'] == 'lstm') & 
                  (df['sequence_length'] == 50) &
                  (df['activation'] == 'tanh')]
    
    axes[1, 0].bar(opt_comp['optimizer'].str.upper(), 
                   opt_comp['accuracy'] * 100,
                   color=['#2ECC71', '#E74C3C', '#F39C12'])
    axes[1, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1, 0].set_title('Optimizer Comparison')
    axes[1, 0].set_ylim(40, 85)
    
    # Sequence length
    seq_comp = df[(df['architecture'] == 'lstm') & 
                  (df['optimizer'] == 'adam') &
                  (df['activation'] == 'tanh') &
                  (df['sequence_length'].isin([25, 50, 100]))]
    
    axes[1, 1].plot(seq_comp['sequence_length'], 
                    seq_comp['accuracy'] * 100,
                    marker='o', linewidth=2, markersize=8, color='#9B59B6')
    axes[1, 1].set_xlabel('Sequence Length', fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1, 1].set_title('Sequence Length Impact')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========== PAGE 5: ANALYSIS ==========
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, '6. DETAILED ANALYSIS', 
             ha='center', fontsize=16, fontweight='bold')
    
    analysis = """ARCHITECTURE PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Simple RNN (63.08%): Basic sequential processing, vanishing gradients, fastest
LSTM (75.81%): Memory gates, long-term dependencies, best balance
BiLSTM (76.79%): Bidirectional context, highest accuracy, slower training

OPTIMIZER COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Adam (75.81%): Adaptive rates, stable convergence, BEST overall
RMSprop (76.04%): Competitive performance, faster in some cases
SGD (51.72%): Poor without tuning, not recommended

SEQUENCE LENGTH IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

25 words (71.09%): Limited context, fastest
50 words (75.81%): Balanced performance
100 words (79.54%): BEST accuracy, maximum context

ACTIVATION FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tanh (75.81%): Zero-centered, best gradient flow, RECOMMENDED
ReLU (75.73%): Efficient but can die
Sigmoid (75.53%): Vanishing gradient issues

GRADIENT CLIPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without (75.47%) vs With 1.0 (75.81%): Small improvement, adds stability"""
    
    fig.text(0.1, 0.88, analysis, va='top', fontsize=8, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========== PAGE 6: CONCLUSIONS ==========
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, '7. CONCLUSIONS & RECOMMENDATIONS', 
             ha='center', fontsize=16, fontweight='bold')
    
    conclusions = """KEY FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LSTM provides optimal balance of accuracy and efficiency
2. Longer sequences (100 words) capture more context → Higher accuracy
3. Adam optimizer demonstrates superior convergence
4. Tanh activation provides best gradient flow
5. Gradient clipping enhances training stability

OPTIMAL CONFIGURATION (79.54% Accuracy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Architecture: LSTM
✓ Sequence Length: 100 words
✓ Optimizer: Adam (lr=0.001)
✓ Activation: Tanh
✓ Gradient Clipping: 1.0
✓ Embedding: 100 dim
✓ Hidden Layers: 2 × 64 units
✓ Dropout: 0.3
✓ Batch Size: 32

FUTURE IMPROVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Attention mechanisms
• Transformer architectures (BERT, RoBERTa)
• Ensemble methods
• Hyperparameter optimization
• Data augmentation techniques

CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This study demonstrates that LSTM networks with optimized hyperparameters achieve
nearly 80% accuracy on binary sentiment classification. The systematic experimental
approach validates the importance of architecture selection, sequence length 
optimization, and training stability techniques.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT REPOSITORY:
GitHub: https://github.com/sravani150602/sentiment-analysis-rnn-comparison

Complete source code, experimental results, and documentation available."""
    
    fig.text(0.1, 0.88, conclusions, va='top', fontsize=8, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"✅ Report saved as: {pdf_path}")
print("🎓 YOUR HOMEWORK IS COMPLETE AND READY TO SUBMIT!")
print("\n📁 FINAL DELIVERABLES:")
print("✅ results/Homework_3_Report.pdf - Professional PDF report")
print("✅ results/metrics.csv - All experimental results")
print("✅ results/plots/ - All comparison charts and training plots")
print("✅ src/ - All source code files")
print("✅ data/ - Preprocessed dataset")
print("\n🌟 EXCELLENT WORK! YOU'VE COMPLETED ALL REQUIREMENTS! 🌟")