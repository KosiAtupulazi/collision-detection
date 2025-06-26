from fastapi import FastAPI, HTTPException, Request
import os
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.mount("/static", StaticFiles(directory="demo_videos"), name="static")


def get_video_path(video_name: str, request: Request) -> str:
    source = os.getenv("VIDEO_SOURCE", "local")

    if source == "local":
        # # video_dir = os.path.join(os.path.dirname(__file__), "demo_videos")
        # # local_path = os.path.join(video_dir, video_name)
        # # #local_path = os.path.join("app", "demo_videos", video_name)

        # if not os.path.exists(local_path):
        #       raise HTTPException(status_code=404, detail="local path not found")
        # return local_path

        return f"{request.base_url}static/{video_name}"

    elif source == "gcs":
        bucket = os.getenv("GCP_BUCKET_NAME", "place_holder_bucket")

        return f"https://storage.googleapis.com/{bucket}/demo_videos/{video_name}"
    else:
        raise HTTPException(status_code=404, detail="gcs path not found")


@app.get("/")
def root():
    return {"Message": "Collision Detection page is working"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/demo")
def predict_demo(request: Request):

    demo_videos = ["crash_vid1.mp4", "crash_vid2.mp4", "vid3.mp4"]
    results = []

    for video_name in demo_videos:
        try:
            # video_path = get_video_path(video_name)
            video_url = get_video_path(video_name, request)
            if "crash" in video_name:
                prediction = "collision"
                confidence = 0.93
                timestamp = "00:03 - 00:05"
            else:
                prediction = "no collision"
                confidence = 0.95
                timestamp = "-"

            # confidence = 0.93 if prediction == "collision" else 0.12  # dummy value
            # timestamp = "00:03 - 00:05" if prediction == "collision" else "—"

            results.append(
                {
                    "video_name": video_name,
                    "video_path": video_url,
                    "prediction": prediction,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
            )

        except HTTPException as e:
            # raise HTTPException (status_code=404, detail="video not found")
            results.append(
                {"Video Name": video_name, "Status Code": "404", "Detail": e.detail}
            )

    return JSONResponse(content=results)
