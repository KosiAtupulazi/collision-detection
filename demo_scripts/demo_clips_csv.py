import pandas as pd

# Load your original full test CSV
input_csv = "/home/atupulazi/personal_projects/collision-detection/final_dataset/test.csv"
output_csv = "/home/atupulazi/personal_projects/collision-detection/demo_scripts/demo_csv_copy.csv"

df = pd.read_csv(input_csv)

# Add a new column 'clip_name' based on padded ID
df["clip_name"] = df["id"].apply(lambda x: f"{int(x):05d}_clip.npy")

# Save it
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
