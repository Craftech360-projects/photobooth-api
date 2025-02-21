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
<<<<<<< HEAD
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
=======
import logging
from fastapi.middleware.cors import CORSMiddleware
>>>>>>> 29f4956de4b0bfc1afb896b5b0740c099be08a12

# Initialize FastAPI app
app = FastAPI()

<<<<<<< HEAD
=======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Initialize GFPGAN for face enhancement
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

>>>>>>> 29f4956de4b0bfc1afb896b5b0740c099be08a12
# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

<<<<<<< HEAD
# Initialize FaceAnalysis
print("Initializing FaceAnalysis...")
try:
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("FaceAnalysis initialized successfully.")
except Exception as e:
    print(f"Error initializing FaceAnalysis: {e}")
    raise
=======
# Initialize SQLite database
DATABASE = "user.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            result_image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()  # Initialize the database on app startup

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    logging.info("Starting face swap...")
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)
    
    logging.info(f"Number of faces detected in source image: {len(facesimg1)}")
    logging.info(f"Number of faces detected in target image: {len(facesimg2)}")
>>>>>>> 29f4956de4b0bfc1afb896b5b0740c099be08a12

# Initialize Face Swapper
print("Loading Face Swapper model...")
try:
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False, download_zip=False)
    print("Face Swapper model loaded successfully.")
except Exception as e:
    print(f"Error loading Face Swapper model: {e}")
    raise

# Initialize GFPGAN for face enhancement
print("Loading GFPGAN model...")
try:
    gfpganer = GFPGANer(
        model_path='models/GFPGANv1.4.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2
    )
    print("GFPGAN model loaded successfully.")
except Exception as e:
    print(f"Error loading GFPGAN model: {e}")
    raise


def load_image(file_path):
    """Load an image using OpenCV."""
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {file_path}")
        raise HTTPException(status_code=500, detail=f"Failed to load image: {file_path}")
    print(f"Image loaded: {file_path} (shape: {image.shape})")
    return image


def save_image(image, folder):
    """Save an image and return the file path."""
    file_name = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(folder, file_name)
    cv2.imwrite(file_path, image)
    print(f"Image saved: {file_path}")
    return file_path


def single_face_swap(source_img, target_img, face_app, swapper):
    """Perform a single face swap."""
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


def two_face_swap(source_img, target_img, face_app, swapper):
    """Perform a two-face swap."""
    print("Starting two-face swap...")
    faces_src = face_app.get(source_img)
    faces_tgt = face_app.get(target_img)

    print(f"Faces detected in source image: {len(faces_src)}")
    print(f"Faces detected in target image: {len(faces_tgt)}")

    if len(faces_src) < 2 or len(faces_tgt) < 2:
        print("Less than two faces detected in one or both images.")
        return None

    # Swap the first face from source with the first face from target
    face_src1 = faces_src[0]
    face_tgt1 = faces_tgt[1]
    img_swapped1 = swapper.get(source_img, face_src1, face_tgt1, paste_back=True)

    # Swap the second face from source with the second face from target
    face_src2 = faces_src[1]
    face_tgt2 = faces_tgt[0]
    img_swapped2 = swapper.get(img_swapped1, face_src2, face_tgt2, paste_back=True)

    print("Two-face swap completed.")
    return img_swapped2


def enhance_face(image):
    """Enhance a face using GFPGAN."""
    print("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)

    if isinstance(restored_img, np.ndarray):
        print("Face enhancement completed.")
        return restored_img
    else:
        print("Face enhancement failed.")
        raise HTTPException(status_code=500, detail="Face enhancement failed.")


@app.post("/api/swap-face/")
<<<<<<< HEAD
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
    """API endpoint for face swapping."""
    try:
        # Save uploaded files
        src_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
        tgt_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)
        with open(src_path, "wb") as buffer:
            shutil.copyfileobj(sourceImage.file, buffer)
        with open(tgt_path, "wb") as buffer:
            shutil.copyfileobj(targetImage.file, buffer)
=======
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...), name: str = File(...), email: str = File(...)):
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)
    print('userDetails',name,email)
>>>>>>> 29f4956de4b0bfc1afb896b5b0740c099be08a12

        # Load images
        source_img = load_image(src_path)
        target_img = load_image(tgt_path)

        # Perform face swap (use single or two-face swap as needed)
        swapped_img = single_face_swap(source_img, target_img, face_app, swapper)
        # swapped_img = two_face_swap(source_img, target_img, face_app, swapper)  # Uncomment if needed

        if swapped_img is None:
            raise HTTPException(status_code=400, detail="Face swap failed.")

        # Enhance the swapped image
        enhanced_img = enhance_face(swapped_img)

        # Save the enhanced image
        result_path = save_image(enhanced_img, RESULT_FOLDER)

        return FileResponse(result_path)
    except Exception as e:
        print(f"Error during face swap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


<<<<<<< HEAD
if __name__ == "__main__":
=======
    logging.info(f"Enhanced image shape: {enhanced_image.shape}")

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, enhanced_image)

    # Save user data in SQLite
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (name, email, result_image_path)
            VALUES (?, ?, ?)
            """,
            (name, email, result_path),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save user data in database: {e}")
        raise HTTPException(status_code=500, detail="Failed to save user data")

    logging.info(f"User details saved: {name}, {email}, {result_path}")
    logging.info(f"Image saved to: {result_path}")

    return FileResponse(result_path)

# HTTP server
if __name__ == '__main__':
>>>>>>> 29f4956de4b0bfc1afb896b5b0740c099be08a12
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
