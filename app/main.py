from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from PIL import Image
import numpy as np
import cv2
import uuid
import os

from app.utils import generate_caption, get_central_point, run_sam_hq, overlay_mask

app = FastAPI()

# Enable CORS if calling from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/segment-wound/")
async def segment_wound(file: UploadFile = File(...)):
    # Save uploaded file
    uid = str(uuid.uuid4())
    image_path = f"{UPLOAD_DIR}/{uid}.jpg"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    # Step 1: Caption image (BLIP-2)
    caption = generate_caption(image_pil)

    # Step 2: Get marker points (here we use center; can be enhanced)
    points, labels = get_central_point(image_np)

    # Step 3: Segment with SAM-HQ2
    mask = run_sam_hq(image_np, points, labels)

    # Step 4: Overlay mask
    segmented_img_np = overlay_mask(image_np, mask)

    # Save segmented image
    result_path = f"{RESULT_DIR}/{uid}_segmented.jpg"
    cv2.imwrite(result_path, cv2.cvtColor(segmented_img_np, cv2.COLOR_RGB2BGR))

    return JSONResponse({
        "caption": caption,
        "result_image": result_path
    })
