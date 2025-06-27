import torch
import torch.nn.functional as F
from model import build_model_r3d_18
from dataset import ClipDataset
from torch.utils.data import DataLoader
import pandas as pd
import json
from tqdm import tqdm
import os

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv_path = "/home/atupulazi/personal_projects/collision-detection/final_dataset/test.csv"
model_path = "checkpoints/r3d_18_best.pth"
output_json = "demo_predictions.json"

# ========== LOAD DATA ==========
label_df = pd.read_csv(test_csv_path)
test_dataset = ClipDataset(test_csv_path, split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ========== LOAD MODEL ==========
model = build_model_r3d_18()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# ========== INFERENCE ==========
results = []

with torch.no_grad():
    for i, (clip, label) in enumerate(tqdm(test_loader)):
        clip, label = clip.to(device), label.to(device)
        output = model(clip)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        # Map to original CSV info
        csv_row = label_df.iloc[i]
        event_time = csv_row.get('time_of_event')
        event_timestamp = None

        if pd.notna(event_time):
            mins = int(event_time) // 60
            secs = event_time % 60
            event_timestamp = f"{mins:02}:{secs:05.2f}"  # e.g., "00:20.02"

        results.append({
            "index": i,
            "true_label": int(label.item()),
            "predicted_label": int(pred.item()),
            "confidence": float(confidence.item()),
            "correct": int(pred.item()) == int(label.item()),
            "time_of_event": event_timestamp
        })

# ========== FILTER & SAVE TOP 5 ==========
correct_results = [r for r in results if r["correct"]]
top_5 = sorted(correct_results, key=lambda x: x["confidence"], reverse=True)[:5]

with open(output_json, "w") as f:
    json.dump(top_5, f, indent=2)

print(f"Saved top 5 confident correct predictions to {output_json}")
