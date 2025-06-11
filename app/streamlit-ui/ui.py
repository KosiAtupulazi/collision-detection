import streamlit as st
import requests

API_URL = "http://fastapi:8002/demo"

st.title("🚗 Collision Detection Demo")
st.caption("Simulated predictions on demo dashcam videos.")

response = requests.get(API_URL)

if response.status_code == 200:
    videos = response.json()

    for video in videos:
        if "error" in video:
            st.error(f"{video['video_name']}: {video['error']}")
        else:
            st.subheader(f"🎥 {video['video_name']}")
            st.text(f"Video path: {video['video_path']}")
            st.video(video["video_path"])
            st.markdown(f"**Prediction:** `{video['prediction']}`")
            st.markdown(f"**Confidence:** `{round(video['confidence']*100)}%`")
            st.markdown(f"**Timestamp:** `{video['timestamp']}`")
            st.divider()
else:
    st.error("Failed to fetch demo video results from FastAPI.")
