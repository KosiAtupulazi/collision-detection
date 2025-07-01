import pandas as pd
import os
import shutil

# Load CSV
df = pd.read_csv("data/demo_predictions_with_paths.csv")

# Filter top confident crash and no_crash separately
top_crash = df[df["label"] == "crash"].sort_values(by="confidence", ascending=False).head(10)
top_no_crash = df[df["label"] == "no_crash"].sort_values(by="confidence", ascending=False).head(10)

# Combine into one balanced DataFrame
balanced_top = pd.concat([top_crash, top_no_crash]).reset_index(drop=True)

# Create demo_videos folder
os.makedirs("app/demo_videos", exist_ok=True)

# Copy videos and update paths
for i, row in balanced_top.iterrows():
    original_path = os.path.join("..", row["video_path"])  # adjust as needed
    filename = os.path.basename(row["video_path"])
    target_path = os.path.join("app/demo_videos", filename)

    if os.path.exists(original_path):
        shutil.copy(original_path, target_path)
        balanced_top.at[i, "video_path"] = f"demo_videos/{filename}"
    else:
        print(f"Missing: {original_path}")

# Save final CSV
balanced_top.to_csv("app/demo_balanced14.csv", index=False)
