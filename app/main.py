from fastapi import FastAPI
import os
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

def get_video_path(video_name : str) -> str:
      source = os.getenv("VIDEO_SOURCE", "local")

      if source == "local":
            local_path = os.path.join("app", "demo_videos", video_name)

            if not os.path.exists(local_path):
                  raise HTTPException(status_code=404, detail="local path not found")
            return local_path
      
      elif source == "gcs": 
            bucket = os.getenv("GCP_BUCKET_NAME", "place_holder_bucket")

            return f"https://storage.googleapis.com/{bucket}/demo_videos/{video_name}"
      else:
        raise HTTPException(status_code=404, detail="gcs path not found")


app = FastAPI()

@app.get("/")

def root():
    return {"Message" : "Collision Detection page is working"}

@app.get("/demo")

def predict_demo():

    demo_videos = ["crash_vid1.mp4", "crash_vid2.mp4", "vid3.mp4"]
    results = []

    for video_name in demo_videos: 
        try:
            video_path = get_video_path(video_name)
            if "crash" in video_name:
                prediction = "collision"
            else:
                prediction = "no collision"
            
            results.append({
                "Video Name": video_name,
                "Video Path": video_path,
                "Prediction": prediction
            })

        except HTTPException as e:
            #raise HTTPException (status_code=404, detail="video not found")
            results.append({
                 "Video Name": video_name,
                 "Status Code": "404",
                 "Detail": e.detail
            })

        
    return JSONResponse(content=results)