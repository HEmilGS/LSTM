import pickle

SELECTED_CLASSES = {
    "ApplyEyeMakeup": 0,
    "Clap": 19,
    "JumpingJack": 46,
    "PushUps": 82,
    "Walking": 99
}

input_pkl = "data/raw/ucf101_2d.pkl"
output_pkl = "data/ucf101_skeleton_5cls.pkl"

with open(input_pkl, "rb") as f:
    data = pickle.load(f)

filtered_annotations = []

for ann in data["annotations"]:
    if ann["label"] in SELECTED_CLASSES.values():
        filtered_annotations.append(ann)

data["annotations"] = filtered_annotations

print(f"Total videos originales: {len(filtered_annotations)}")

label_map = {v: i for i, v in enumerate(SELECTED_CLASSES.values())}

for ann in data["annotations"]:
    ann["label"] = label_map[ann["label"]]

with open(output_pkl, "wb") as f:
    pickle.dump(data, f)

print("Nuevo PKL guardado:", output_pkl)
print("Classes mapeadas:", label_map)
