import os
import cv2
import json
import base64
import torch
from moviepy.editor import VideoFileClip
from unsloth import FastLanguageModel
from PIL import Image
from io import BytesIO
from openai import OpenAI

# Load API keys from environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load LLaMA 3.2 Vision model
max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    
)
FastLanguageModel.for_inference(model)

print("✅ LLaMA Vision model loaded")

def extract_frames(video_path, out_dir=".tmp/frames", frame_rate=5):
    """Extract frames from video"""
    os.makedirs(out_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    count = 0
    for t in range(0, int(clip.duration), frame_rate):
        frame = clip.get_frame(t)
        frame_path = os.path.join(out_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        count += 1
    clip.close()
    return sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")])

def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """Extract audio"""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="mp3")
    clip.close()
    return audio_path

def transcribe_audio(audio_path):
    """Whisper transcription"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def caption_frame(frame_path, prompt="Describe this video frame in detail."):
    """Caption a single frame"""
    try:
        image = Image.open(frame_path).convert("RGB")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
        caption = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        return f"(failed: {str(e)})"

def caption_frames(frame_dir):
    """Caption all frames"""
    captions = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        print(f"➡️ Captioning {frame_path}...")
        captions.append({"frame": frame_path, "caption": caption_frame(frame_path)})
    return captions

def summarize_video(video_path, out_json="outputs/result.json"):
    """Full pipeline: Frames + Whisper + GPT summary"""
    os.makedirs("outputs", exist_ok=True)

    # Extract frames and captions
    frames = extract_frames(video_path, frame_rate=5)
    frame_captions = caption_frames(".tmp/frames")

    # Audio transcription
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)

    combined_text = " ".join([c["caption"] for c in frame_captions if not c["caption"].startswith("(failed:")])
    final_prompt = f"""
    Here are video frame descriptions:
    {combined_text}

    Here is the audio transcript:
    {transcript}

    Please generate a clear and concise video summary combining both visual and audio information.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=400,
    )
    summary = response.choices[0].message.content

    result = {
        "frames": frame_captions,
        "transcript": transcript,
        "summary": summary,
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    return result
