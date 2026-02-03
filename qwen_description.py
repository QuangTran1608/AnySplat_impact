import numpy as np
import cv2
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def load_video(video_path, sampling=16):
    cap = cv2.VideoCapture(video_path)

    ret_frames = []
    frame_idx = 0

    while cap.isOpened():
        print("cap openeed")
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx % sampling) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret_frames.append(frame)

        frame_idx += 1
    
    cap.release()
    return np.stack(ret_frames)
 
model_name = "Qwen/Qwen2-VL-7B-Instruct"
 
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
 
processor = AutoProcessor.from_pretrained(model_name)
 
 
video_path = "C:/west_project/AnySplat_impact/video_output_path/rgb.mp4"
frames = load_video(video_path)
 
question = "Here is the view of an office. How many people do you see in the video? If there are people, what are they doing?"
 
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": question},
        ],
    }
]
 
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
 
inputs = processor(
    text=[text],
    videos=[frames],
    padding=True,
    return_tensors="pt"
).to(model.device)
 
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )
 
output_text = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]
 
print("Model answer:\n", output_text)