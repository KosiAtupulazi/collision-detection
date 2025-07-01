import streamlit as st
import pandas as pd
import os
import shutil
import base64

# --- Load prediction data with paths ---
csv_path = "demo_top15.csv"
df = pd.read_csv(csv_path)

label_map = {0: "no_crash", 1: "crash"}
df["ground_truth"] = df["ground_truth"].map(label_map)
df["prediction"] = df["prediction"].map(label_map)

#st.markdown("### Model Predictions on All Test Clips")

# metrics_path = "/home/atupulazi/personal_projects/collision-detection/src/test_metrics/test_metrics20250625_171400.csv"  # adjust path if needed

# if os.path.exists(metrics_path):
#     metrics_df = pd.read_csv(metrics_path)
#     latest_metrics = metrics_df.iloc[-1]  # get most recent row

#     st.markdown("### 🧪 Model Test Metrics")
#     st.markdown(f"""
#     **Accuracy:** `{latest_metrics['Accuracy']:.4f}`  
#     **Precision:** `{latest_metrics['Precision']:.4f}`  
#     **Recall:** `{latest_metrics['Recall']:.4f}`  
#     **F1 Score:** `{latest_metrics['F1']:.4f}`  
#     """)
# else:
#     st.warning("Metrics file not found. Please check the path.")

def load_video_as_base64(path):
    with open(path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded = base64.b64encode(video_bytes).decode()
        return f"data:video/mp4;base64,{encoded}"

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📽️ Demo", "🧠 Model Info", "📄 Project README"])


with tab1:
    st.title("🚗 Collision Detection Demo 🚗")

    top10 = df.sort_values(by="confidence", ascending=False).head(15)


    for i, row in top10.iterrows():

        # video_full_path = row["video_path"]
        # st.write("Trying to load:", video_full_path)
        # st.write("Exists:", os.path.exists(video_full_path))

        

        st.markdown("---")
        st.markdown(f"**Clip:** `{row['clip_name']}`")

        video_full_path = row["video_path"]
        # if os.path.exists(video_full_path):
        #     st.video(video_full_path)
        # else:
        #     st.warning(f"Video not found: {row['video_path']}")

        if os.path.exists(video_full_path):
            video_data_url = load_video_as_base64(video_full_path)
            st.video(video_data_url)
        else:
             st.warning(f"Video not found: {video_full_path}")

        st.markdown(f"""
        **Ground Truth:** `{row['ground_truth']}`  
        **Model Prediction:** `{row['prediction']}`  
        **Confidence:** `{row['confidence']:.2f}`  
        **Time of Event:** `{row.get('time_of_event', 'N/A')}`
        """)


with tab2:
    st.header("🧠 Model Overview")
    st.markdown("""
    **Model:** `R3D_18`  
    **Architecture:** 3D CNN  
    **Input:** `16-frame video clips (1.6 sec)`  
    **Output:** Binary classification → `crash` or `no_crash`  
    **Trained on:** `Labeled Nexar Dashcam Clips`

    **Evaluation Metrics:**  
    - Accuracy: `75.62%`  
    - Precision: `75.73%`  
    - Recall: `75.62%`  
    - F1 Score: `75.60%`  
    """)


with tab3:
    st.header("📄 Project README")
    st.markdown("""
    ### 🎯 Objective  
    Predict whether a crash is occurring in short dashcam video clips using a deep learning model.

    ### 📦 Dataset  
    - Source: Nexar dashcam dataset  
    - Each clip: `16 frames (~1.6 seconds)`  
    - Labeled as `crash` or `no_crash`  
    - Balanced sample from real-world driving footage

    ### 🔨 Approach  
    - Extract video frames from clips  
    - Feed into 3D ResNet-18 model (`R3D_18`)  
    - Output binary classification  
    - Evaluate using accuracy, precision, recall, and F1

    ### 🛠 Tools Used  
    - PyTorch  
    - OpenCV  
    - Streamlit (for UI)  
    - Google Cloud (for storage/deployment)

    ### ✅ Results  
    - Accuracy: 75.6% on test set  
    - Good separation between crash/no_crash classes  
    - Real-time playback + UI for predictions

    ### 🚀 Next Steps  
    - Extend to crash forecasting (predicting before it happens)  
    - Add temporal attention or LSTM head  
    - Deploy full MLOps pipeline

    ### 🙋🏽‍♀️ Author  
    Kosi Atupulazi | Master's Student, Major in Artificial Intelligence | University of Texas at San Antonio (UTSA)  
    """)
