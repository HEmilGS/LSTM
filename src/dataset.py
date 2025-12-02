import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, pkl_data, class_map, num_frames=50):
        self.items = pkl_data["annotations"]
        self.class_map = class_map
        self.num_frames = num_frames

    def __len__(self):
        return len(self.items)

    def sample_frames(self, pose, total_frames):
        idx = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        return pose[idx]  

    def __getitem__(self, idx):
        item = self.items[idx]

        kp = item["keypoint"]   
        T = item["total_frames"]
        label = int(item["label"])

        if kp.ndim == 4:
            kp = kp[0]    
        elif kp.ndim == 3:
            kp = kp
        else:
            kp = np.zeros((self.num_frames, 17, 2))
            label = 0

        real_length = min(kp.shape[0], self.num_frames)

        if kp.shape[0] >= self.num_frames:
            kp = self.sample_frames(kp, kp.shape[0])
            real_length = self.num_frames
        else:
            # Padding con el último frame
            last = kp[-1]
            pad = np.repeat(last[None, :, :], self.num_frames - kp.shape[0], axis=0)
            kp = np.concatenate([kp, pad], axis=0)

        # Normalización
        h, w = item["img_shape"]
        kp[..., 0] /= w
        kp[..., 1] /= h

        kp = torch.tensor(kp, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        real_length = torch.tensor(real_length, dtype=torch.long)

        # Devolvemos también la longitud real para pack_padded_sequence
        return kp, label, real_length