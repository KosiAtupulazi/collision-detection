import pandas as pd

df = pd.read_csv("frames/train/train_clip_labels.csv")

# Print any row with clip_name matching 00318
print(df[df['clip_name'].str.contains("00318", case=False)])
