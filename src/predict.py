import torch
import pickle
import numpy as np
from model import SimpleLSTM, ImprovedLSTM, BaselineFC

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


def preprocess_skeleton(kp, img_shape, num_frames=50):
    """Preprocesa un skeleton para predicción"""
    if kp.ndim == 4:
        kp = kp[0]  # Solo primera persona
    elif kp.ndim != 3:
        raise ValueError(f"Dimensiones incorrectas: {kp.ndim}, esperaba 3 o 4")
    
    # Guardar longitud real
    real_length = min(kp.shape[0], num_frames)
    
    if kp.shape[0] >= num_frames:
        idx = np.linspace(0, kp.shape[0] - 1, num_frames).astype(int)
        kp = kp[idx]
    else:
        last = kp[-1]
        pad = np.repeat(last[None, :, :], num_frames - kp.shape[0], axis=0)
        kp = np.concatenate([kp, pad], axis=0)
    
    # Normalizar
    h, w = img_shape
    kp[..., 0] /= w
    kp[..., 1] /= h
    
    kp = torch.tensor(kp, dtype=torch.float32).unsqueeze(0)  
    real_length = torch.tensor([real_length], dtype=torch.long)
    
    return kp, real_length


def predict_by_index(model, pkl_data, video_idx):
    """Predice la clase de un video dado su índice"""
    if video_idx < 0 or video_idx >= len(pkl_data["annotations"]):
        raise ValueError(f"Índice fuera de rango: {video_idx}. Debe estar entre 0 y {len(pkl_data['annotations'])-1}")
    
    item = pkl_data["annotations"][video_idx]
    
    kp = item["keypoint"]
    img_shape = item["img_shape"]
    true_label = item["label"]
    frame_dir = item.get("frame_dir", "N/A")
    
    kp, lengths = preprocess_skeleton(kp, img_shape)
    kp, lengths = kp.to(DEVICE), lengths.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        logits = model(kp, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
    
    return {
        "video_idx": video_idx,
        "frame_dir": frame_dir,
        "true_label": true_label,
        "true_class": CLASS_NAMES[true_label],
        "pred_label": pred_class,
        "pred_class": CLASS_NAMES[pred_class],
        "confidence": probs[pred_class].item() * 100,
        "all_probs": {CLASS_NAMES[i]: probs[i].item() * 100 for i in range(5)}
    }


def interactive_predict(model_name="improved_lstm", checkpoint="best"):
    """Modo interactivo para hacer predicciones"""
    checkpoint_path = f"results/{model_name}_{checkpoint}.pth"
    
    print(f"Cargando modelo: {model_name} ({checkpoint})")
    model = load_model(model_name, checkpoint_path)
    
    print("Cargando dataset...")
    with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
        pkl_data = pickle.load(f)
    
    print(f"\nDataset cargado: {len(pkl_data['annotations'])} videos")
    print(f"Clases disponibles: {', '.join(CLASS_NAMES)}\n")
    
    while True:
        print("\n" + "="*60)
        print("Opciones:")
        print("  1. Predecir por índice")
        print("  2. Predecir varios aleatorios")
        print("  3. Salir")
        print("="*60)
        
        choice = input("\nElige una opción: ").strip()
        
        if choice == "1":
            try:
                idx = int(input(f"Índice del video (0-{len(pkl_data['annotations'])-1}): "))
                result = predict_by_index(model, pkl_data, idx)
                print_result(result)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            n = int(input("¿Cuántos videos aleatorios? "))
            indices = np.random.choice(len(pkl_data["annotations"]), size=min(n, len(pkl_data["annotations"])), replace=False)
            
            correct = 0
            for idx in indices:
                result = predict_by_index(model, pkl_data, idx)
                print_result(result, compact=True)
                if result["pred_label"] == result["true_label"]:
                    correct += 1
            
            print(f"\nPrecisión en muestra aleatoria: {correct}/{len(indices)} = {correct/len(indices)*100:.2f}%")
        
        elif choice == "4":
            break
        
        else:
            print("Opción inválida")


def print_result(result, compact=False):
    """Imprime el resultado de una predicción"""
    if compact:
        correct = "✓" if result["pred_label"] == result["true_label"] else "✗"
        print(f"{correct} Video {result['video_idx']:4d}: Real={result['true_class']:15s} | Pred={result['pred_class']:15s} ({result['confidence']:.1f}%)")
    else:
        print("\n" + "-"*60)
        print(f"Video índice: {result['video_idx']}")
        print(f"Video nombre: {result['frame_dir']}")
        print(f"Clase real:   {result['true_class']} (label {result['true_label']})")
        print(f"Predicción:   {result['pred_class']} (label {result['pred_label']})")
        print(f"Confianza:    {result['confidence']:.2f}%")
        print("\nProbabilidades por clase:")
        for class_name, prob in result['all_probs'].items():
            bar = "█" * int(prob / 2)
            print(f"  {class_name:15s}: {prob:5.2f}% {bar}")
        
        if result["pred_label"] == result["true_label"]:
            print("\n✓ Predicción CORRECTA")
        else:
            print("\n✗ Predicción INCORRECTA")
        print("-"*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="improved_lstm",
                        choices=["baseline_fc", "simple_lstm", "improved_lstm"])
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--index", type=int, default=None, help="Índice del video a predecir")
    
    args = parser.parse_args()
    
    if args.index is not None:
        # Modo no interactivo
        checkpoint_path = f"results/{args.model}_{args.checkpoint}.pth"
        model = load_model(args.model, checkpoint_path)
        
        with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
            pkl_data = pickle.load(f)
        
        if args.index is not None:
            result = predict_by_index(model, pkl_data, args.index)
        
        print_result(result)
    else:
        # Modo interactivo
        interactive_predict(args.model, args.checkpoint)