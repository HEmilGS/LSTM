import torch
import pickle
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from model import SimpleLSTM
from tqdm import tqdm

CLASS_MAP = {
    "ApplyEyeMakeup": 0,
    "Clapping": 1,
    "JumpingJack": 2,
    "PushUps": 3,
    "Walking": 4
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
        pkl_data = pickle.load(f)

    allowed_labels = set(CLASS_MAP.values())
    annotations = [a for a in pkl_data["annotations"] if a["label"] in allowed_labels]

    print("Total anotaciones:", len(pkl_data["annotations"]))
    print("Anotaciones filtradas:", len(annotations))

    pkl_data["annotations"] = annotations

    dataset = SkeletonDataset(
        pkl_data,
        class_map=CLASS_MAP,
        num_frames=50
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleLSTM(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load("results/model_5cls.pth", map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for seq, label in tqdm(loader):
            seq, label = seq.to(DEVICE), label.to(DEVICE)

            pred = model(seq)
            pred_class = pred.argmax(dim=1)

            correct += (pred_class == label).sum().item()
            total += label.size(0)

    print(f"\nAccuracy total: {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    evaluate()
