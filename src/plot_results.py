import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_history(model_names=None):
    """Grafica la historia de entrenamiento de uno o varios modelos"""
    if model_names is None:
        model_names = ["baseline_fc", "simple_lstm", "improved_lstm", "bilstm", "gru"]
    
    if isinstance(model_names, str):
        model_names = [model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for model_name in model_names:
        history_path = f"results/{model_name}_history.json"
        
        if not os.path.exists(history_path):
            print(f"Historia no encontrada: {history_path}")
            continue
        
        with open(history_path, "r") as f:
            history = json.load(f)
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Train loss
        axes[0].plot(epochs, history["train_loss"], marker='o', label=model_name, linewidth=2)
        
        # Val loss
        axes[1].plot(epochs, history["val_loss"], marker='o', label=model_name, linewidth=2)
        
        # Val accuracy
        axes[2].plot(epochs, history["val_acc"], marker='o', label=model_name, linewidth=2)
    
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Validation Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/training_comparison.png", dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: results/training_comparison.png")
    plt.show()


def plot_model_comparison():
    """Compara el mejor accuracy de cada modelo"""
    models = ["baseline_fc", "simple_lstm", "improved_lstm", "bilstm", "gru"]
    model_labels = ["Baseline FC", "Simple LSTM", "Improved LSTM", "BiLSTM", "GRU"]
    
    best_accs = []
    final_accs = []
    
    for model_name in models:
        history_path = f"results/{model_name}_history.json"
        
        if not os.path.exists(history_path):
            best_accs.append(0)
            final_accs.append(0)
            continue
        
        with open(history_path, "r") as f:
            history = json.load(f)
        
        best_accs.append(max(history["val_acc"]))
        final_accs.append(history["val_acc"][-1])
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, best_accs, width, label='Mejor Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Accuracy Final', alpha=0.8)
    
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Comparación de Modelos - Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: results/model_comparison.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="both", 
                        choices=["history", "comparison", "both"])
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Nombres de modelos para graficar (solo para history)")
    
    args = parser.parse_args()
    
    if args.type in ["history", "both"]:
        plot_training_history(args.models)
    
    if args.type in ["comparison", "both"]:
        plot_model_comparison()