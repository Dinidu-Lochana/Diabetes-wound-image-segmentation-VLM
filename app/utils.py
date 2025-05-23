import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything_hq import sam_model_registry, SamPredictor
import cv2

# Load BLIP-2 model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM-HQ2 model
checkpoint_path = "models/sam_hq_vit_b.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

def generate_caption(image_pil):
    inputs = processor(images=image_pil, return_tensors="pt").to(blip_model.device)
    output = blip_model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def get_central_point(image_np):
    h, w, _ = image_np.shape
    return np.array([[w // 2, h // 2]]), np.array([1])  # label 1 = foreground

def run_sam_hq(image_np, point_coords, point_labels):
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    return masks[0]  # Best mask

def overlay_mask(image_np, mask):
    result = image_np.copy()
    result[mask] = [0, 255, 0]  # Green overlay
    return result
