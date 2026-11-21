import os
import csv
import time
import datetime
import cv2  # OpenCV for camera control
import dotenv
import glob
import threading
from pydub.utils import mediainfo
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_sa
from flask import Flask, jsonify, make_response
from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from playsound import playsound



# === CONFIGURATION ===
dotenv.load_dotenv("keys.env")
GEMINI_API_KEY = dotenv.get_key("keys.env", "GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash" # Updated to a modern, capable model
RECORDING_DURATION = 5  # Seconds
CSV_FILE_NAME = "log.csv"
MAX_RETRIES = 10
POLL_INTERVAL = 1  # Check every 1s until ACTIVE
RETRY_DELAY = 10    # Wait before retrying after a 500 error

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# === INITIALIZATION ===
app = Flask(__name__)

# Ensure necessary folders exist
os.makedirs("recordings", exist_ok=True)

# === HELPER FUNCTIONS ===
def record_video(filename: str, duration: int, warmup_frames: int = 10):
    """Records a video from the USB camera for a set duration (seconds)."""
    print(f"🎥 Starting {duration}s recording...")

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # <-- safer default to 0
    if not cap.isOpened():
        print("🚨 Error: Could not open camera.")
        return False

    # Force resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 🔥 Warmup: discard unstable startup frames
    for _ in range(warmup_frames):
        cap.read()
    time.sleep(0.5)  # give sensor time to settle
    print(f"✅ Camera warmed up with {warmup_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print("🚨 Error: VideoWriter failed to open.")
        cap.release()
        return False

    # 🔄 Ensure we can actually grab a frame before starting
    ret, frame = cap.read()
    retries = 0
    while not ret and retries < 10:
        print("⚠️ No frame yet, retrying...")
        time.sleep(0.1)
        ret, frame = cap.read()
        retries += 1
    if not ret:
        print("🚨 Camera never returned a valid frame, aborting recording.")
        cap.release()
        out.release()
        return False

    # 🎬 Recording loop
    start_time = time.time()
    frame_count = 0
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            print("⚠️ Frame grab failed mid-recording, stopping.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print(f"🚨 No frames written! Video file {filename} may be invalid.")
        return False

    print(f"✅ Recording saved to {filename} with {frame_count} frames")
    return True


def analyze_emotion_from_video(video_path: str):
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
                contents=["Please describe what emotions the human is expressing in 1-2 sentences.", video_file]
            )
        except Exception as e:
            if "500" in str(e):
                print(f"⚠️ Gemini 500 INTERNAL for {os.path.basename(video_path)}")
                return "ERROR: Gemini 500 INTERNAL"
            else:
                print(f"⚠️ Unexpected Gemini error: {e}")
                return f"ERROR: {e}"

        # Validate Gemini response
        if not response.text:
            finish_reason = (
                response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
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
        # Always clean up uploaded files
        if video_file:
            try:
                client.files.delete(name=video_file.name)
                print(f"🗑️ Deleted uploaded file: {video_file.name}")
            except Exception as e:
                print(f"⚠️ Could not delete file {video_file.name}: {e}")

def play_audio_background(audio_file: str):
    """Play audio in a separate thread using pydub/simpleaudio."""
    audio = AudioSegment.from_file(audio_file, format="mp3")
    playback = play_sa(audio)  # async
    return playback, audio.duration_seconds


def play_audio_with_recording(audio_file: str, label: str, pre_roll: float = 1.0):
    """Record video while audio plays, with warmup + pre-roll buffer."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recordings/{label}_{timestamp}.mp4"

    # Load audio metadata
    audio = AudioSegment.from_file(audio_file, format="mp3")
    audio_duration = audio.duration_seconds

    # Total recording = pre-roll + audio + 5s tail
    total_duration = pre_roll + audio_duration + 5

    print(f"🎥 Recording {total_duration:.2f}s (pre-roll {pre_roll:.1f}s) while playing {audio_file}")

    # Start recording (this runs in the main thread)
    record_thread = threading.Thread(
        target=record_video, args=(video_filename, total_duration)
    )
    record_thread.start()

    # Let camera capture pre-roll before audio starts
    time.sleep(pre_roll)

    # Now play audio asynchronously
    playback = play_sa(audio)

    # Wait for audio + recording
    playback.wait_done()
    record_thread.join()

    print(f"✅ Finished playing {audio_file} and recording {video_filename}")
    return video_filename




def generate_emotion_aware_apology(emotion_context: str) -> str:
    """Generates an apology using the emotion context."""
    print("✍️ Generating emotion-aware apology...")
    prompt = f"""
    Adapt the current apology to better match the detected emotional context in 1-2 sentences.
    Current apology: Apologies for the delay; here are your items.
    Context: {emotion_context}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        if not response.text:
            finish_reason = (
                response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            )
            print(f"⚠️ No content generated for apology. Finish Reason: {finish_reason}")
            return f"ERROR: No response (Reason: {finish_reason})"
        return response.text.strip()
    except Exception as e:
        if "500" in str(e):
            return "ERROR: Gemini 500 INTERNAL"
        else:
            print(f"⚠️ Unexpected Gemini error during apology generation: {e}")
            return f"ERROR: {e}"

def log_to_csv(emotion: str, apology: str):
    """Appends the results to a CSV file."""
    file_exists = os.path.isfile(CSV_FILE_NAME)
    with open(CSV_FILE_NAME, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Detected Emotion", "Generated Apology"])
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, emotion, apology])
    print(f"📝 Results logged to {CSV_FILE_NAME}")

def generate_audio(text: str, output_filename: str):
    """Generates an MP3 file from text using Google Cloud Text-to-Speech."""
    try:
        # Load credentials from keys.env
        dotenv.load_dotenv("keys.env")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Initialize TTS client
        client = texttospeech.TextToSpeechClient()

        # Input text
        ssml_text = f"<speak>{text}<break time='250ms'/></speak>"
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        # Configure a natural neural voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O"
        )

        # Set audio output format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # 0.25 ~ 4.0
            pitch=0.0           # Adjust tone if needed
        )

        # Perform synthesis
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # ✅ Check if the file exists and remove it first
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Save output MP3
        
        with open(output_filename, "wb") as out:
            out.write(response.audio_content)

        print("🔊 Google TTS audio generated successfully.")

    except Exception as e:
        print(f"🚨 Google TTS failed: {e}")


# === FLASK ROUTES ===

@app.route('/')
def index():
    """Root endpoint for checking server status."""
    return jsonify({"status": "running", "message": "Temi API server is live!"})

@app.route('/EA_process', methods=['POST'])
def EA_process():
    """Endpoint that Temi will call to start the whole process."""
    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    
    # 1. Get video filename
     # Find all mp4 files in the directory
    video_files = glob.glob(os.path.join("recordings", "*.mp4"))
    if not video_files:
        return None  # No video found
    # Sort files by modification time, latest first
    video_filename = max(video_files, key=os.path.getmtime)

    # 2. Analyze emotion with retries
    emotion = ""
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔄 Attempting emotion analysis {attempt} of {MAX_RETRIES}")
        emotion = analyze_emotion_from_video(video_filename)

        if "ERROR" not in emotion:
            success = True
            break
        else:
            print(f"⚠️ Emotion analysis failed → Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not success:
        print(f"🚨 FAILED emotion analysis after {MAX_RETRIES} attempts → {video_filename}")
        return jsonify({"status": "error", "message": emotion}), 500

    # 3. Generate apology with retries --- THIS BLOCK IS NOW FIXED ---
    apology = ""
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔄 Attempting apology generation {attempt} of {MAX_RETRIES}")
        apology = generate_emotion_aware_apology(emotion)

        if "ERROR" not in apology:
            success = True
            break
        else:
            print(f"⚠️ Apology generation failed → Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not success:
        print(f"🚨 FAILED to generate apology after {MAX_RETRIES} attempts.")
        return jsonify({"status": "error", "message": apology}), 500

    # 4. Generate audio
    generate_audio(apology, "tailored_apology.mp3")

    # 5. Log results
    log_to_csv(emotion, apology)
    
    # 6. Sending the response immediately
    print(f"📦 Returning JSON: emotion='{emotion}', apology='{apology}'")
    response = make_response(jsonify({
        "status": "success",
        "emotion": emotion,
        "apology": apology
    }), 200)
    response.headers["Content-Type"] = "application/json"
    print("--- PROCESS COMPLETE ---")
    return response

@app.route('/NE_process', methods=['POST'])
def NE_process():
    """Endpoint that Temi will call to start the whole process."""
    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    
    # 1. Get video filename
     # Find all mp4 files in the directory
    video_files = glob.glob(os.path.join("recordings", "*.mp4"))
    if not video_files:
        return None  # No video found
    # Sort files by modification time, latest first
    video_filename = max(video_files, key=os.path.getmtime)

    # 2. Analyze emotion with retries
    emotion = ""
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔄 Attempting emotion analysis {attempt} of {MAX_RETRIES}")
        emotion = analyze_emotion_from_video(video_filename)

        if "ERROR" not in emotion:
            success = True
            break
        else:
            print(f"⚠️ Emotion analysis failed → Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not success:
        print(f"🚨 FAILED emotion analysis after {MAX_RETRIES} attempts → {video_filename}")
        return jsonify({"status": "error", "message": emotion}), 500

    # 3. Log results
    apology = "Apologies for the delay; here are your items."
    log_to_csv(emotion, apology)
    
    # 4. Sending the response immediately
    print(f"📦 Returning JSON: emotion='{emotion}', apology='{apology}'")
    response = make_response(jsonify({
        "status": "success",
        "emotion": emotion,
        "apology": apology
    }), 200)
    response.headers["Content-Type"] = "application/json"
    print("--- PROCESS COMPLETE ---")
    return response

@app.route('/play_control', methods=['Post'])
def play_control():
    """Endpoint that Temi will call to start the whole process."""

    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    play_audio_with_recording("delivery_message.mp3", "control")
    print("--- PROCESS COMPLETE ---")

    return jsonify({"status": "success"}), 200
    

@app.route('/play_NE', methods=['POST'])
def play_NE():
    """Endpoint that Temi will call to start the whole process."""

    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    play_audio_with_recording("generic_apology.mp3", "NE")
    print("--- PROCESS COMPLETE ---")
    
    return jsonify({"status": "success"}), 200

@app.route('/play_EA', methods=['POST'])
def play_EA():
    """Endpoint that Temi will call to start the whole process."""

    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    play_audio_with_recording("tailored_apology.mp3", "EA")
    print("--- PROCESS COMPLETE ---")
    
    return jsonify({"status": "success"}), 200
    
@app.route('/control_process', methods=['POST'])
def control_process():
    """Endpoint that Temi will call to start the whole process."""
    print("\n--- TRIGGER RECEIVED FROM TEMI ---")
    
    # 1. Get video filename
     # Find all mp4 files in the directory
    video_files = glob.glob(os.path.join("recordings", "*.mp4"))
    if not video_files:
        return None  # No video found
    # Sort files by modification time, latest first
    video_filename = max(video_files, key=os.path.getmtime)

    # 2. Analyze emotion with retries
    emotion = ""
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔄 Attempting emotion analysis {attempt} of {MAX_RETRIES}")
        emotion = analyze_emotion_from_video(video_filename)

        if "ERROR" not in emotion:
            success = True
            break
        else:
            print(f"⚠️ Emotion analysis failed → Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not success:
        print(f"🚨 FAILED emotion analysis after {MAX_RETRIES} attempts → {video_filename}")
        return jsonify({"status": "error", "message": emotion}), 500

    apology = "CONTROL CASE: NO APOLOGY GENERATED"

    # 3. Log results
    log_to_csv(emotion, apology)
    
    # 4. Sending the response immediately
    print(f"📦 Returning JSON: emotion='{emotion}', apology='{apology}'")
    response = make_response(jsonify({
        "status": "success",
        "emotion": emotion,
        "apology": apology
    }), 200)
    response.headers["Content-Type"] = "application/json"
    print("--- PROCESS COMPLETE ---")
    return response

if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible on your local network
    app.run(host='0.0.0.0', port=5000)