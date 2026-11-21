import os
import torch
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress deepface's TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# ========= NEW MODEL IMPORTS =========
from deepface import DeepFace
import torch.hub
from ultralytics import YOLO
from torchvision.io import read_video
import torchvision.transforms as T

# ========= CONFIG =========
VIDEO_DIR = os.getenv("STUDY1_VIDEO_DIR", "videos")  # Study 1 clips from the Bridges dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= MODEL LOADING =========

object_model = YOLO("yolov9c.pt")





# ========= HELPER FUNCTIONS =========

def analyze_emotion_deepface(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            cap.release()
            return "no frames in video"
        middle_frame_idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret: return "could not read frame"
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and len(result) > 0:
            return result[0]['dominant_emotion']
        else:
            return "no face detected"
    except Exception:
        return "error analyzing emotion"


def detect_objects(model, video_path, frame_sample_rate=5):
    detected_objects = set()
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_sample_rate == 0:
                results = model(frame, verbose=False, device=DEVICE, conf=0.8)
                for result in results:
                    for label_id in result.boxes.cls.tolist():
                        cls_name = model.names[int(label_id)]
                        detected_objects.add(cls_name)
            frame_idx += 1
        cap.release()
    except Exception:
        return ["error detecting objects"]
    return list(detected_objects)

# ========= MAIN PIPELINE =========
def analyze_video(video_path):
    emotion_label = analyze_emotion_deepface(video_path)
    objects_list = detect_objects(object_model, str(video_path))
    emotion_str = f"{emotion_label.lower()}"
    objects_str = ", ".join(objects_list) if objects_list else ""
    total_parts = [emotion_str]
    if objects_str:
        total_parts.append(f"{objects_str}")
    total_str = " ".join(total_parts)
    return {
        "video_filename": video_path.name,
        "emotion": emotion_str,
        "objects": objects_str,
        "total": total_str
    }

# ========= MAIN EXECUTION =========
def main():
    video_dir = Path(VIDEO_DIR)
    video_files = list(video_dir.glob("*.mp4"))
    all_results = []

    if not video_files:
        print(f"No .mp4 videos found in {VIDEO_DIR}")
        return

    for video_path in tqdm(video_files, desc="Processing videos with specialized CNNs"):
        try:
            result = analyze_video(video_path)
            if result:
                all_results.append(result)
        except Exception as e:
            # --- THE SECOND CRITICAL FIX ---
            # Use repr(e) to get a guaranteed string representation of the error
            print(f"\nFATAL ERROR processing {video_path.name}: {repr(e)}. Skipping this video.")
            all_results.append({
                "video_filename": video_path.name,
                "emotion": "CRASHED",
                "objects": "CRASHED", "total": "CRASHED"
            })

    if not all_results:
        print("No results were generated.")
        return
    output_filename = "cnn_results_specialized.csv"
    headers = all_results[0].keys()
    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} results to {output_filename}")

if __name__ == "__main__":
    main()