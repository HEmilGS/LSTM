import torch
from torch.utils.data import DataLoader
from model import SimpleLSTM
from dataset import SkeletonDataset
from tqdm import tqdm
import pickle
import os

CLASS_MAP = {
    "ApplyEyeMakeup": 0,
    "Clapping": 1,
    "JumpingJack": 2,
    "PushUps": 3,
    "Walking": 4
}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/ucf101_skeleton_5cls.pkl", "rb") as f:
        pkl_data = pickle.load(f)

    allowed_labels = set(CLASS_MAP.values())
    annotations = [a for a in pkl_data["annotations"] if a["label"] in allowed_labels]

    print("Videos encontrados:", len(annotations))
    
    pkl_data["annotations"] = annotations

    dataset = SkeletonDataset(
        pkl_data,
        class_map=CLASS_MAP,
        num_frames=50
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleLSTM(num_classes=5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(15):
        total_loss = 0
        model.train()

        for seq, label in tqdm(loader):
            seq, label = seq.to(device), label.to(device)

            optim.zero_grad()
            pred = model(seq)
            loss = loss_fn(pred, label)
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model_5cls.pth")
    print("Modelo guardado en results/model_5cls.pth")


if __name__ == "__main__":
    train()
