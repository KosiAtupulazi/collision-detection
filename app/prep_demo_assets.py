import pandas as pd
import os
import shutil

# Load CSV
df = pd.read_csv("data/demo_predictions_with_paths.csv")
top10 = df.sort_values(by="confidence", ascending=False).head(10)

# Create videos folder
os.makedirs("app/videos", exist_ok=True)

# Copy videos and update paths
for i, row in top10.iterrows():
    original = row["video_path"]
    source = os.path.join("..", original)  # adjust as needed
    dest_name = os.path.basename(original)
    dest_path = os.path.join("app/videos", dest_name)
    shutil.copy(source, dest_path)
    df.loc[i, "video_path"] = f"videos/{dest_name}"

# Save modified CSV with only top 10 + fixed paths
df.head(10).to_csv("app/demo_top10.csv", index=False)
