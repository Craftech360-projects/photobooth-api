thsi is my cleint now


import sqlite3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
import uuid
from gfpgan import GFPGANer
import numpy as np
from PIL import Image
import logging
from fastapi.middleware.cors import CORSMiddleware
import urllib.request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Model paths and URLs
inswapper_model_path = 'models/inswapper_128.onnx'
inswapper_model_url = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'

gfpgan_model_path = 'models/GFPGANv1.4.pth'
gfpgan_model_url = 'https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth'

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Download inswapper model if not present
if not os.path.exists(inswapper_model_path):
    print(f"Downloading {inswapper_model_path} ...")
    urllib.request.urlretrieve(inswapper_model_url, inswapper_model_path)
    print("Download complete.")

# Download GFPGAN model if not present
if not os.path.exists(gfpgan_model_path):
    print(f"Downloading {gfpgan_model_path} ...")
    urllib.request.urlretrieve(gfpgan_model_url, gfpgan_model_path)
    print("Download complete.")

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def create_4x6_canvas(image, width=695, height=1074):
    """
    Resize image to specified dimensions and center it on a 4x6 canvas (1200x1800 pixels)
    """
    # Resize image to target dimensions
    resized = cv2.resize(image, (width, height))
    
    # Create white canvas of 4x6 size (1200x1800 pixels)
    canvas = np.full((1800, 1200, 3), 255, dtype=np.uint8)
    
    # Calculate position to center the image
    x_offset = (1200 - width) // 2
    y_offset = (1800 - height) // 2
    
    # Place the image in the center of canvas
    canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized
    
    return canvas


def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    logging.info("Starting face swap..., ")
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)
    
    logging.info(f"Number of faces detected in source image: {len(facesimg1)}")
    logging.info(f"Number of faces detected in target image: {len(facesimg2)}")

    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected
    
    face1 = facesimg1[0]
    face2 = facesimg2[0]

    img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
    logging.info("Face swap completed.")
    return img1_swapped

def enhance_face(image):
    logging.info("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    
    logging.info(f"Type of restored_img: {type(restored_img)}")
    if isinstance(restored_img, Image.Image):
        restored_img = np.array(restored_img)
    logging.info(f"Type after conversion (if any): {type(restored_img)}")
    if isinstance(restored_img, np.ndarray):
        logging.info("Face enhancement completed.")
        return restored_img
    else:
        raise ValueError("Enhanced image is not a valid numpy array")

# Function to generate the target image using AI
def generate_target_image(prompt=None):
    clipdrop_api_key = '287e9967894a412606ce693e3ada3e65054ec9120b001c7968d1631a8209dae4be082e26afb8084cd9a13c1390d26dfa'
    predefined_prompts_str = "photorealistic concept art, high quality digital art, cinematic, hyperrealism, photorealism, Nikon D850, 8K., sharp focus, emitting diodes, artillery, motherboard, by pascal blanche rutkowski repin artstation hyperrealism painting concept art of detailed character design matte painting, 4 k resolution"

    all_prompts = predefined_prompts_str
    if prompt:
        all_prompts = prompt

    clipdrop_url = 'https://clipdrop-api.co/text-to-image/v1'
    headers = {
        'x-api-key': clipdrop_api_key,
        'accept': 'image/webp',
        'x-clipdrop-width': '400',
        'x-clipdrop-height': '600',
    }
    data = {
        'prompt': (None, all_prompts, 'text/plain')
    }
    response = requests.post(clipdrop_url, files=data, headers=headers)

    if response.ok:
        with open('static/target_image.webp', 'wb') as f:
            f.write(response.content)
    else:
        print(f"Clipdrop API error: {response.status_code} - {response.text}")

@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...), name: str = File(...), email: str = File(...)):
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)
    print('userDetails',name,email)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)
    with open(img2_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    sourceImage_cv = cv2.imread(img1_path)
    targetImage_cv = cv2.imread(img2_path)

    if sourceImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read source image with OpenCV: {img1_path}")
    if targetImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read target image with OpenCV: {img2_path}")

    logging.info(f"Source image shape: {sourceImage_cv.shape}")
    logging.info(f"Target image shape: {targetImage_cv.shape}")

    swapped_image = simple_face_swap(sourceImage_cv, targetImage_cv, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    logging.info(f"Swapped image shape: {swapped_image.shape}")

    enhanced_image = enhance_face(swapped_image)

    # Add this line after enhancement
    final_image = create_4x6_canvas(enhanced_image)

    logging.info(f"Final image shape: {final_image.shape}")

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, final_image)

    return FileResponse(result_path)

# HTTP server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)


next message I will sendfrontend code


