import os
import uuid
import shutil
import cv2
import requests
import numpy as np
import sqlite3
import warnings
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis
import insightface
from gfpgan import GFPGANer

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Directory Setup ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# --- Model Paths and URLs ---
INSWAPPER_PATH = os.path.join(MODEL_FOLDER, 'inswapper_128.onnx')
INSWAPPER_URL = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'

GFPGAN_PATH = os.path.join(MODEL_FOLDER, 'GFPGANv1.4.pth')
GFPGAN_URL = 'https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth'

# --- Helper: Download if Missing ---
def download_file_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"{os.path.basename(path)} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {path}: {e}")
            raise

# --- Download Required Models ---
download_file_if_missing(INSWAPPER_URL, INSWAPPER_PATH)
download_file_if_missing(GFPGAN_URL, GFPGAN_PATH)

# --- Load FaceAnalysis ---
print("Initializing FaceAnalysis...")
try:
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("FaceAnalysis initialized successfully.")
except Exception as e:
    print(f"Error initializing FaceAnalysis: {e}")
    raise

# --- Load Face Swapper ---
print("Loading Face Swapper model...")
try:
    swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, download=False, download_zip=False)
    print("Face Swapper model loaded successfully.")
except Exception as e:
    print(f"Error loading Face Swapper model: {e}")
    raise

# --- Load GFPGAN Model ---
print("Loading GFPGAN model...")
try:
    gfpganer = GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=1,
        arch='clean',
        channel_multiplier=2
    )
    print("GFPGAN model loaded successfully.")
except Exception as e:
    print(f"Error loading GFPGAN model: {e}")
    raise

# --- Utility: Load Image ---
def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {file_path}")
        raise HTTPException(status_code=500, detail=f"Failed to load image: {file_path}")
    print(f"Image loaded: {file_path} (shape: {image.shape})")
    return image

# --- Utility: Save Image ---
def save_image(image, folder):
    file_name = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(folder, file_name)
    cv2.imwrite(file_path, image)
    print(f"Image saved: {file_path}")
    return file_path

# --- Utility: Single Face Swap ---
def single_face_swap(source_img, target_img, face_app, swapper):
    print("Starting single face swap...")
    faces_src = face_app.get(source_img)
    faces_tgt = face_app.get(target_img)
    print(f"Faces detected in source image: {len(faces_src)}")
    print(f"Faces detected in target image: {len(faces_tgt)}")

    if not faces_src or not faces_tgt:
        print("No faces detected in one or both images.")
        return None

    face_src = faces_src[0]
    face_tgt = faces_tgt[0]
    swapped_img = swapper.get(source_img, face_src, face_tgt, paste_back=True)
    print("Single face swap completed.")
    return swapped_img

# --- Utility: Enhance Face ---
def enhance_face(image):
    print("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    if isinstance(restored_img, np.ndarray):
        print("Face enhancement completed.")
        return restored_img
    else:
        print("Face enhancement failed.")
        raise HTTPException(status_code=500, detail="Face enhancement failed.")

# --- Face Swap Endpoint ---
@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
    try:
        # Save uploaded files
        src_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_source.jpg")
        tgt_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_target.jpg")

        with open(src_path, "wb") as f:
            shutil.copyfileobj(sourceImage.file, f)
        with open(tgt_path, "wb") as f:
            shutil.copyfileobj(targetImage.file, f)

        # Load images (ensure correct order)
        source_img = load_image(src_path)
        target_img = load_image(tgt_path)

        # Perform swap
        swapped_img = single_face_swap(source_img, target_img, face_app, swapper)
        if swapped_img is None:
            raise HTTPException(status_code=400, detail="Face swap failed.")

        # Enhance face
        enhanced_img = enhance_face(swapped_img)

        # Save result
        result_path = save_image(enhanced_img, RESULT_FOLDER)
        return FileResponse(result_path)

    except Exception as e:
        print(f"Error during face swap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Run App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
