from pathlib import Path
import torch
import os
import sys
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
import cv2

# Load the model from Hugging Face
video_urls = ["rtsp://192.168.1.190:554/stream/main",
              "rtsp://192.168.1.192:554/stream/main"]
model = AnySplat.from_pretrained("lhjiang/anysplat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Load and preprocess example images (replace with your own image paths)
# image_names = []
# for i in range(50):
#     image_names.append(f"/home/lauretta/quang/AnySplat/test_images_1/{i*2}.jpg")
#caps = []
#for video_string in video_urls:
#    caps.append(cv2.VideoCapture(video_string))

image_paths = ["output_images/1.jpg", "output_images/2.jpg"]
# frames = [cap.read()[1] for cap in caps]
images = [process_image(frame) for frame in image_paths]
images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
b, v, _, h, w = images.shape
print("running inference now")
# Run Inference
begin = time()
gaussians, pred_context_pose = model.inference((images+1)*0.5)
print("running postprocess now")
pred_all_extrinsic = pred_context_pose['extrinsic']
pred_all_intrinsic = pred_context_pose['intrinsic']
end = time()
print(f"Inference time: {end - begin:.2f} seconds")
save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, "video_output_path", model.decoder)
end2 = time()
print(f"Total time (inference + rendering): {end2 - begin:.2f} seconds")
time.sleep(5)
