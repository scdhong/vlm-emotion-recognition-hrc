import subprocess
import os
import statistics
import re

def run_benchmark(script_name):
    """
    Runs a python script and extracts both 'Cloud Processing' (elapsed) 
    and 'Inference' times from its output.
    """
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found in current directory.")
        return None

    print(f"▶️  STARTING BENCHMARK: {script_name}")
    print(f"   (Running full pipeline and capturing per-video times. Please wait...)\n")
    
    # Run the script and capture the output (stdout)
    # text=True ensures we get a string; errors="ignore" helps bypass emoji encoding issues
    process = subprocess.run(["python", script_name], capture_output=True, text=True, errors="ignore")
    
    # Check if the script crashed
    if process.returncode != 0:
        print(f"❌ Script failed! Error output:\n{process.stderr}")
        return None

    processing_times = []
    inference_times = []

    for line in process.stdout.splitlines():
        # 1. Capture the 'elapsed' variable (Cloud Processing Time)
        # Matches: "File ACTIVE after 1.5s"
        proc_match = re.search(r"ACTIVE after\s*([\d\.]+)s", line)
        if proc_match:
            processing_times.append(float(proc_match.group(1)))

        # 2. Capture the 'INFERENCE_TIME'
        # Matches: "INFERENCE_TIME: 2.45"
        inf_match = re.search(r"INFERENCE_TIME:\s*([\d\.]+)", line)
        if inf_match:
            inference_times.append(float(inf_match.group(1)))

    if not inference_times:
        print(f"   ⚠️ WARNING: No data found in {script_name}.")
        return None
        
    print(f"✅ FINISHED: {script_name} ({len(inference_times)} videos processed)")
    return {
        "processing": processing_times,
        "inference": inference_times
    }

def print_metric_row(label, timings):
    """Helper function to calculate and format a row of statistics."""
    if not timings:
        print(f"  ↳ {label:<30} | (No data collected)")
        return
        
    mean_t = statistics.mean(timings)
    min_t = min(timings)
    max_t = max(timings)
    std_t = statistics.stdev(timings) if len(timings) > 1 else 0.0
    
    print(f"  ↳ {label:<30} | {mean_t:>6.2f}s | {min_t:>6.2f}s | {max_t:>6.2f}s | {std_t:>6.2f}s")

if __name__ == "__main__":
    scripts_to_test = [
        "gemini-flash-response-script.py", 
        "gemini_response_script.py"
    ]
    
    results = {}

    print("="*85)
    print("HRC STUDY 1: DUAL-LATENCY BENCHMARK (PROCESSING VS. INFERENCE)")
    print("="*85)

    for script in scripts_to_test:
        data = run_benchmark(script)
        if data:
            results[script] = data

    # Summary Report
    if results:
        print("\n" + "="*85)
        print("FINAL SUMMARY FOR REVIEWER RESPONSE")
        print("="*85)
        print(f"{'Metric / Script Name':<35} | {'Mean':<7} | {'Min':<7} | {'Max':<7} | {'Std Dev':<7}")
        print("-" * 85)
        
        for name, data in results.items():
            print(f"{name}:")
            # Report the 'elapsed' variable data
            print_metric_row("Video Uploading times", data["processing"])
            # Report the Inference data
            print_metric_row("Model Inference times", data["inference"])
            print("-" * 85)
        
        print("="*85)