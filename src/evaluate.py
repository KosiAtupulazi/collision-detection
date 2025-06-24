# loads model + val/test set, evaluates accuracy

import torch
from model import build_model_r3d_18  # replace with actual class name
from dataset import ClipDataset
from torch.utils.data import DataLoader
import pandas as pd

# ========== SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_csv_path = "/home/atupulazi/personal_projects/collision-detection/frames/test/test_clip_labels.csv"  # <-- update if different
test_dataset = ClipDataset(test_csv_path, split='test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load model
model = build_model_r3d_18()  # Replace with your model class
model.load_state_dict(torch.load("/home/atupulazi/personal_projects/collision-detection/src/checkpoints/r3d18_final.pth"))  # Your .pth file here
model.to(device)
model.eval()

# ========== TEST LOOP ==========
correct = 0
total = 0

with torch.no_grad():
    for clips, labels in test_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
