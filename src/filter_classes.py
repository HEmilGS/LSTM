import pickle

SELECTED_CLASSES = {
    "ApplyEyeMakeup": 0,
    "Clapping": 19,  
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

print(f"Total videos filtrados: {len(filtered_annotations)}")

# Mapear labels originales a 0-4
label_map = {v: i for i, v in enumerate(SELECTED_CLASSES.values())}
print(f"Mapeo de labels: {label_map}")

for ann in data["annotations"]:
    ann["label"] = label_map[ann["label"]]

# Verificar distribución de clases
class_counts = {}
for ann in data["annotations"]:
    label = ann["label"]
    class_counts[label] = class_counts.get(label, 0) + 1

print("\nDistribución de clases:")
for cls_name, cls_id in sorted([(k, label_map[v]) for k, v in SELECTED_CLASSES.items()], key=lambda x: x[1]):
    print(f"  {cls_name} (label {cls_id}): {class_counts.get(cls_id, 0)} videos")

with open(output_pkl, "wb") as f:
    pickle.dump(data, f)

print(f"\nNuevo PKL guardado: {output_pkl}")