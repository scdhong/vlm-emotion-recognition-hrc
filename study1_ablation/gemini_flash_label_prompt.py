import os
import csv
import time
import dotenv
from google import genai
from google.api_core.exceptions import GoogleAPICallError

# === CONFIGURATION ===
dotenv.load_dotenv("keys.env")
GEMINI_API_KEY = dotenv.get_key("keys.env", "GEMINI_API_KEY")
VIDEO_FOLDER_PATH = os.getenv("STUDY1_VIDEO_DIR", "videos")  # Study 1 clips from the Bridges dataset
CSV_FOLDER_PATH = "."  # results are written next to this script
CSV_FILE_NAME = "ablation_gemini_flash_analysis_results.csv"
PROMPT = "Analyze this video. First, choose the single most dominant human emotion from the following list: [happy, sad, angry, neutral, surprise, fear, disgust]. Second, list up to the four most dominant subjects and objects in the scene relevant to the interaction. Format your answer ONLY as: emotion_label, object1, object2, object3, object4"
MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 10
POLL_INTERVAL = 1  # Check every 1s until ACTIVE
RETRY_DELAY = 10    # Wait before retrying after a 500 error

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_video(video_path):
    """
    Uploads a video, waits until ACTIVE, analyzes it using Gemini,
    and deletes the uploaded file.
    """
    video_file = None
    try:
        # Upload video
        video_file = client.files.upload(file=video_path)
        print(f"📤 Uploaded: {os.path.basename(video_path)}, initial state: {video_file.state.name}")

        # Wait for the video to finish processing
        start_time = time.time()
        while True:
            video_file = client.files.get(name=video_file.name)
            state = video_file.state.name
            if state == "ACTIVE":
                elapsed = time.time() - start_time
                print(f"✅ File ACTIVE after {elapsed:.1f}s, analyzing video...")
                break
            elif state == "FAILED":
                print(f"❌ Processing failed for {os.path.basename(video_path)}")
                return "ERROR: Processing failed"
            print(f"⏳ File still processing ({state}), retrying in {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)

        # Send video + prompt to Gemini
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[PROMPT, video_file]
            )
        except Exception as e:
            # Catch Gemini server-side 500 errors
            if "500" in str(e):
                print(f"⚠️ Gemini 500 INTERNAL for {os.path.basename(video_path)}")
                return "ERROR: Gemini 500 INTERNAL"
            else:
                print(f"⚠️ Unexpected Gemini error: {e}")
                return f"ERROR: {e}"

        # Validate Gemini response
        if not response.text:
            finish_reason = (
                response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            )
            print(f"⚠️ No content generated. Finish Reason: {finish_reason}")
            return f"ERROR: No response (Reason: {finish_reason})"

        print(f"📝 Gemini Response: {response.text.strip()}")
        return response.text.strip()

    except GoogleAPICallError as e:
        print(f"🚨 API Error: {e}")
        return f"ERROR: {e}"
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return f"ERROR: {e}"
    finally:
        # Always clean up uploaded files if ACTIVE
        if video_file and video_file.state.name == "ACTIVE":
            client.files.delete(name=video_file.name)
            print(f"🗑️ Deleted uploaded file: {video_file.name}")

def main():
    """
    Loops through all videos, analyzes them, retries failed ones, and writes results to CSV.
    """
    csv_file_path = os.path.join(CSV_FOLDER_PATH, CSV_FILE_NAME)
    video_files = [f for f in os.listdir(VIDEO_FOLDER_PATH) if f.endswith(".mp4")]

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Video Filename", "Emotion Description"])  # CSV header

        for video_file in video_files:
            video_path = os.path.join(VIDEO_FOLDER_PATH, video_file)
            print(f"\n🎥 Processing video: {video_file}")

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"🔄 Attempt {attempt} of {MAX_RETRIES}")
                emotion_description = analyze_video(video_path)

                if "ERROR" not in emotion_description:
                    success = True
                    break
                else:
                    print(f"⚠️ Gemini failed for {video_file} → Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)

            if not success:
                print(f"🚨 FAILED after {MAX_RETRIES} attempts → {video_file}")

            # Save result to CSV regardless of success/failure
            csv_writer.writerow([video_file, emotion_description])
            print(f"✅ Finished: {video_file}")

    print(f"\n🎉 Analysis complete. Results saved to: {csv_file_path}")

if __name__ == "__main__":
    main()
