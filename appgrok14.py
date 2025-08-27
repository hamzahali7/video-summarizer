from dotenv import load_dotenv
load_dotenv()
print("Script started.")
from flask import Flask, request, jsonify
import whisper
import yt_dlp
import os
import uuid
import glob
from groq import Groq
import psutil  # Added for memory monitoring
import time    # Added to track processing times
import gc      # Added for memory cleanup
import sys     # For recursion limit adjustment

print("Imports loaded.")
app = Flask(__name__)
print("Checking initial memory usage...")
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
print(f"Initial memory usage: {initial_memory:.2f} MB")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Initializing Groq API...")
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Groq API initialized.")
except Exception as e:
    print(f"Groq API initialization failed: {str(e)}")
    raise

print("Setting up routes...")

def download_audio(youtube_url):
    video_id = str(uuid.uuid4())
    output_template = os.path.join(DOWNLOAD_DIR, f"{video_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading audio with template: {output_template}")
        ydl.download([youtube_url])

    downloaded_files = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}*.mp3"))
    if not downloaded_files:
        raise FileNotFoundError(f"Download failed. No .mp3 file matching: {video_id}*.mp3")

    return downloaded_files[0]

def summarize_with_groq(transcript):
    print("Initializing Groq summarization...")
    try:
        prompt = f"""
        You are a summarization assistant. Given the following transcript of a video, generate a concise summary strictly in JSON format with:
        - "overview": A 2–3 sentence overview
        - "keyPoints": A list of 5–8 key bullet points
        Ensure the output is valid JSON, enclosed in curly braces, with quoted keys and values.
        Transcript:
        \"\"\"{transcript[:8000]}\"\"\"  # Reduced to 8,000 characters
        """

        print("Sending prompt to Groq...")
        pre_summarization_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage before summarization: {pre_summarization_memory:.2f} MB")
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500  # Reduced from 1000
        )
        print("Received response from Groq")
        raw_response = response.choices[0].message.content
        print("Raw response:", raw_response)

        post_summarization_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage after summarization: {post_summarization_memory:.2f} MB")

        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            summary_text = json_match.group(0)
            try:
                import json
                summary_data = json.loads(summary_text)
                del raw_response  # Release Groq response buffer
                return summary_data
            except json.JSONDecodeError:
                del raw_response  # Release even on error
                return {
                    "overview": "Summary unavailable or improperly formatted.",
                    "keyPoints": []
                }
        else:
            del raw_response  # Release if no match
            return {
                "overview": "Summary unavailable or improperly formatted.",
                "keyPoints": []
            }
    except Exception as e:
        print(f"Summarization failed: {str(e)}")
        return {
            "overview": "Model unavailable. Please check your API key or try a different model.",
            "keyPoints": []
        }

@app.route('/transcribe', methods=['POST'])
def transcribe_and_summarize():
    data = request.get_json()
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({'error': 'Missing YouTube URL'}), 400

    current_memory = process.memory_info().rss / 1024 / 1024
    if current_memory > 300:  # Cap before model load
        return jsonify({'error': 'Memory limit exceeded. Try again later.'}), 503

    print("Loading Whisper model...")
    start_time = time.time()
    model = whisper.load_model("tiny")  # Switched to tiny model
    model_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Memory usage after loading model: {model_memory:.2f} MB")
    print(f"Model loading time: {time.time() - start_time:.2f} seconds")

    try:
        print("Downloading audio...")
        start_time = time.time()
        audio_path = download_audio(youtube_url)
        download_time = time.time() - start_time
        print(f"Download time: {download_time:.2f} seconds")

        print(f"Transcribing file: {audio_path}")
        start_time = time.time()
        transcription_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage before transcription: {transcription_memory:.2f} MB")
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        transcription_time = time.time() - start_time
        post_transcription_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage after transcription: {post_transcription_memory:.2f} MB")
        print(f"Transcription time: {transcription_time:.2f} seconds")

        transcript = result['text']
        print("Summarizing with Groq...")
        start_time = time.time()
        summary = summarize_with_groq(transcript)
        summarization_time = time.time() - start_time
        print(f"Summarization time: {summarization_time:.2f} seconds")
        print(f"Summary: {summary}")

        if post_transcription_memory > 450:  # Increased warning threshold
            print("Warning: Memory usage exceeds 450MB. Consider optimization.")

        # Unload model and clean up
        del model
        del result  # Clear transcription result
        gc.set_threshold(0)  # Force aggressive garbage collection
        gc.collect()
        sys.setrecursionlimit(1000)  # Adjust recursion limit
        final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage after cleanup: {final_memory:.2f} MB")

        response = jsonify({
            'text': transcript,
            'summary': summary,
            'timing': {
                'download': download_time,
                'transcription': transcription_time,
                'summarization': summarization_time
            }
        })
        del transcript  # Delete after response is prepared
        return response
    except Exception as e:
        print(f"Error in transcribe_and_summarize: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Entering main block...")
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)