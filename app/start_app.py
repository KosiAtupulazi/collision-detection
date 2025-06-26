import subprocess
import threading
import time
import requests

def run_fastapi():
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

def wait_for_fastapi():
    while True:
        try:
            r = requests.get("http://localhost:8000/demo")  # or /ping if you add one
            if r.status_code == 200:
                print("FastAPI is running...")
                break
        except Exception:
            pass
        time.sleep(1)  # wait 1 second and try again

def run_streamlit():
    subprocess.run(["streamlit", "run", "ui.py", "--server.port=8080", "--server.address=0.0.0.0"])

# Start FastAPI in a thread
t1 = threading.Thread(target=run_fastapi)
t1.start()

# Wait until FastAPI is fully up
wait_for_fastapi()

# Then start Streamlit
t2 = threading.Thread(target=run_streamlit)
t2.start()

# Wait for both to finish
t1.join()
t2.join()
