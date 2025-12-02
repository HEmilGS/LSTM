import torch
from torch.utils.data import DataLoader, random_split
from model import SimpleLSTM, ImprovedLSTM, BaselineFC
from dataset import SkeletonDataset
from tqdm import tqdm
import pickle
import os
import json

CLASS_MAP = {
    "ApplyEyeMakeup": 0,
    "Clapping": 1,
    "JumpingJack": 2,
    "PushUps": 3,
    "Walking": 4
}


def train_model(model_name="improved_lstm", epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
        pkl_data = pickle.load(f)

    allowed_labels = set(CLASS_MAP.values())
    annotations = [a for a in pkl_data["annotations"] if a["label"] in allowed_labels]

    print(f"Videos encontrados: {len(annotations)}")
    
    pkl_data["annotations"] = annotations

    dataset = SkeletonDataset(
        pkl_data,
        class_map=CLASS_MAP,
        num_frames=50
    )

    # Split train/validation 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {train_size}, Validation: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Seleccionar modelo
    if model_name == "baseline_fc":
        model = BaselineFC(num_classes=5).to(device)
    elif model_name == "simple_lstm":
        model = SimpleLSTM(num_classes=5).to(device)
    elif model_name == "improved_lstm":
        model = ImprovedLSTM(num_classes=5, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    print(f"\nEntrenando modelo: {model_name}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=3
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        for seq, label, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            seq, label, lengths = seq.to(device), label.to(device), lengths.to(device)

            optim.zero_grad()
            pred = model(seq, lengths)
            loss = loss_fn(pred, label)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for seq, label, lengths in val_loader:
                seq, label, lengths = seq.to(device), label.to(device), lengths.to(device)

                pred = model(seq, lengths)
                loss = loss_fn(pred, label)
                val_loss += loss.item()

                pred_class = pred.argmax(dim=1)
                correct += (pred_class == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total * 100

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.2f}%")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), f"results/{model_name}_best.pth")
            print(f"  ✓ Mejor modelo guardado (val_acc={val_acc:.2f}%)")

        scheduler.step(avg_val_loss)

    # Guardar último modelo e historial
    torch.save(model.state_dict(), f"results/{model_name}_last.pth")
    
    with open(f"results/{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEntrenamiento completado!")
    print(f"Mejor validation accuracy: {best_val_acc:.2f}%")
    print(f"Modelos guardados en results/{model_name}_*.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="improved_lstm", 
                        choices=["baseline_fc", "simple_lstm", "improved_lstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )