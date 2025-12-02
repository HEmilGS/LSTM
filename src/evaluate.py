import torch
import pickle
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from model import SimpleLSTM, ImprovedLSTM, BaselineFC
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

CLASS_MAP = {
    "ApplyEyeMakeup": 0,
    "Clapping": 1,
    "JumpingJack": 2,
    "PushUps": 3,
    "Walking": 4
}

CLASS_NAMES = list(CLASS_MAP.keys())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, checkpoint_path):
    """Carga el modelo según el nombre"""
    if model_name == "baseline_fc":
        model = BaselineFC(num_classes=5)
    elif model_name == "simple_lstm":
        model = SimpleLSTM(num_classes=5)
    elif model_name == "improved_lstm":
        model = ImprovedLSTM(num_classes=5, hidden_dim=256, num_layers=2, dropout=0.3)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model.to(DEVICE)


def evaluate(model_name="improved_lstm", checkpoint="best"):
    with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
        pkl_data = pickle.load(f)

    allowed_labels = set(CLASS_MAP.values())
    annotations = [a for a in pkl_data["annotations"] if a["label"] in allowed_labels]

    print(f"Total anotaciones: {len(annotations)}")

    pkl_data["annotations"] = annotations

    dataset = SkeletonDataset(
        pkl_data,
        class_map=CLASS_MAP,
        num_frames=50
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    checkpoint_path = f"results/{model_name}_{checkpoint}.pth"
    print(f"Cargando modelo desde: {checkpoint_path}")
    
    model = load_model(model_name, checkpoint_path)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, label, lengths in tqdm(loader, desc="Evaluando"):
            seq, label, lengths = seq.to(DEVICE), label.to(DEVICE), lengths.to(DEVICE)

            pred = model(seq, lengths)
            pred_class = pred.argmax(dim=1)

            all_preds.extend(pred_class.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Métricas
    accuracy = (all_preds == all_labels).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"Modelo: {model_name} ({checkpoint})")
    print(f"{'='*60}")
    print(f"Accuracy total: {accuracy:.2f}%\n")

    # Reporte de clasificación
    print("Reporte de clasificación:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))

    # Matriz de confusión
    print("\nMatriz de confusión:")
    cm = confusion_matrix(all_labels, all_preds)
    print("Predicho →")
    print(f"{'Real ↓':<20}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:10]:>12}", end="")
    print()
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name[:20]:<20}", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i,j]:>12}", end="")
        print()


def compare_models():
    """Compara todos los modelos entrenados"""
    models = ["baseline_fc", "simple_lstm", "improved_lstm"]
    
    results = []
    
    for model_name in models:
        checkpoint_path = f"results/{model_name}_best.pth"
        try:
            with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
                pkl_data = pickle.load(f)

            allowed_labels = set(CLASS_MAP.values())
            annotations = [a for a in pkl_data["annotations"] if a["label"] in allowed_labels]
            pkl_data["annotations"] = annotations

            dataset = SkeletonDataset(pkl_data, class_map=CLASS_MAP, num_frames=50)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            model = load_model(model_name, checkpoint_path)
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for seq, label, lengths in loader:
                    seq, label, lengths = seq.to(DEVICE), label.to(DEVICE), lengths.to(DEVICE)
                    pred = model(seq, lengths)
                    pred_class = pred.argmax(dim=1)
                    correct += (pred_class == label).sum().item()
                    total += label.size(0)

            acc = correct / total * 100
            results.append((model_name, acc))
            print(f"{model_name:20s}: {acc:.2f}%")
        except FileNotFoundError:
            print(f"{model_name:20s}: Modelo no encontrado")
    
    if results:
        print("\n" + "="*50)
        best_model, best_acc = max(results, key=lambda x: x[1])
        print(f"Mejor modelo: {best_model} con {best_acc:.2f}% accuracy")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="improved_lstm",
                        choices=["baseline_fc", "simple_lstm", "improved_lstm", "all"])
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    
    args = parser.parse_args()
    
    if args.model == "all":
        compare_models()
    else:
        evaluate(args.model, args.checkpoint)