from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import gdown
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
import uuid
from gfpgan import GFPGANer
import numpy as np
from PIL import Image
import logging
import requests
import base64
import json
from pydantic import BaseModel
from supabase import create_client, Client
from typing import List

# Supabase URL and API key (replace with your actual Supabase project URL and API key)
SUPABASE_URL = "https://vfzahhknnbkkuwidgmfa.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmemFoaGtubmJra3V3aWRnbWZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzEzOTQ0MzIsImV4cCI6MjA0Njk3MDQzMn0.1xBDtTprTlchZZ_rKzdSnoHWgYNEF_5sS9pxyOlCuro"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now (use specific origins for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)
class User(BaseModel):
    name: str
    email: str


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown
# URLs for the models
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_url2 = 'https://drive.google.com/uc?id=1jhhRaa66kDupnajlYu2VNEFp7Ekm_yM8'  # Replace with actual GFPGAN model ID

# Output paths for the models
model_output_path = 'inswapper/inswapper_128.onnx'
model_output_path2 = 'models/GFPGANv1.4.pth'

# Ensure directories exist
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
os.makedirs(os.path.dirname(model_output_path2), exist_ok=True)

# Check and download models if they do not exist
if not os.path.exists(model_output_path):
    print(f"{model_output_path} not found. Downloading...")
    gdown.download(model_url, model_output_path, quiet=False)

if not os.path.exists(model_output_path2):
    print(f"{model_output_path2} not found. Downloading...")
    gdown.download(model_url2, model_output_path2, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Initialize GFPGAN for face enhancement
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)



def generate_target_imagePinokio(custom_prompts=None):
   # clipdrop_api_key = '2cac03e37041e25b2d2931b8e4f5dc991d946f651f0062251f6007c50060f482953b8b19cd3fdee27c85b0d2b51bedb9'  # Replace with your actual Clipdrop API key
    predefined_prompts_str = ")) ((full body, clear face)) stands in full growth, photorealism, future,(intricate details:1.12), hdr, (intricate details, hyperdetailed:1.15), (natural skin texture, hyperrealism,  sharp:1.2), futurism, mechanics,((  looking straight at the camera, portrait,)), cinema, cinematic, cinematic light, cinema light, portrait, full face view, high detail, realism, photorealism, natural light, movie, unreal engine, concept artisuals"

# Initialize all_prompts with custom_prompts if available, then append predefined_prompts_str
    if custom_prompts:
     all_prompts = f"(({custom_prompts}))\n{predefined_prompts_str}"
    else:
        all_prompts = predefined_prompts_str

# Print or use the combined prompt
    print(all_prompts)

    pinokio_url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
    

    # data = {
    #     'prompt': (None, all_prompts, 'text/plain')
    # }
    print(all_prompts)
    payload = {
        "alwayson_scripts": {
            "API payload": {
                "args": []
            },
            "AnimateDiff": {
                "args": [{
                    "batch_size": 1,
                    "closed_loop": "R-P",
                    "enable": False,
                    "format": ["GIF", "PNG"],
                    "fps": 8,
                    "freeinit_ds": 0.25,
                    "freeinit_dt": 0.25,
                    "freeinit_enable": False,
                    "freeinit_filter": "butterworth",
                    "freeinit_iters": 3,
                    "interp": "Off",
                    "interp_x": 10,
                    "is_i2i_batch": False,
                    "last_frame": None,
                    "latent_power": 1,
                    "latent_power_last": 1,
                    "latent_scale": 32,
                    "latent_scale_last": 32,
                    "loop_number": 0,
                    "mask_path": "",
                    "model": "",
                    "overlap": -1,
                    "prompt_scheduler": None,
                    "request_id": "",
                    "stride": 1,
                    "video_default": False,
                    "video_length": 0,
                    "video_path": "",
                    "video_source": None
                }]
            },
        },
      #  "prompt": all_prompts,  # Include the combined prompt
        "prompt":custom_prompts ,
        "negative_prompt": "dof, grayscale, black and white, bw, 3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, girl, loli, young, large breasts, red eyes, muscular,badhandsv5-neg, By bad artist -neg (1), monochrome,(EasyNegative:0.8), (worst quality, low quality:1.2), text,  bad quality, watermark, bad anatomy,",
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0.7,
        "height": 600,
        "width": 400,
        "n_iter": 1,
        "sampler_name": "DPM++ 2M",
        "seed": -1,
        "restore_faces": True
    }

    headers = {
        'Content-Type': 'application/json'
    }

    # Send POST request to local API
    try:
        response = requests.post(pinokio_url, json=payload, headers=headers)

        if response.ok:
            print("hereeeeee")
            # print(response.message)
            data = response.json()  # Converts JSON to a Python dictionary
            # print("JSON Response Data:")
            # print(data)
            base64_image_data = data["images"][0]
            # print(base64_image_data)
            # Decode the base64 string into binary data
            image_data = base64.b64decode(base64_image_data)
            # Save the image from the response
            target_image_path = os.path.join(UPLOAD_FOLDER, 'target_image.png')
            with open(target_image_path, 'wb') as f:
                f.write(image_data)
            logging.info(f"Target image generated and saved to {target_image_path}")
            return target_image_path
        else:
            logging.error("Failed to fetch target image from local server")
            raise HTTPException(status_code=500, detail="Failed to fetch target image from local server")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Error connecting to local server")
    

# Function to fetch and save the target image using Clipdrop
def generate_target_imageClipdrop(custom_prompts=None):
    clipdrop_api_key = '2cac03e37041e25b2d2931b8e4f5dc991d946f651f0062251f6007c50060f482953b8b19cd3fdee27c85b0d2b51bedb9'  # Replace with your actual Clipdrop API key
    predefined_prompts_str = "photorealistic concept art, high quality digital art, cinematic, hyperrealism, photorealism, Nikon D850, 8K., sharp focus, emitting diodes, artillery, motherboard, by pascal blanche rutkowski repin artstation hyperrealism painting concept art of detailed character design matte painting, 4 k resolution"

    all_prompts = predefined_prompts_str
    if custom_prompts:
        all_prompts += "\n" + custom_prompts

    clipdrop_url = 'https://clipdrop-api.co/text-to-image/v1'
    headers = {
        'x-api-key': clipdrop_api_key,
        'accept': 'image/webp',
        'x-clipdrop-width': '400',  # Desired width in pixels
        'x-clipdrop-height': '600',  # Desired height in pixels
    }

    data = {
        'prompt': (None, all_prompts, 'text/plain')
    }

    response = requests.post(clipdrop_url, files=data, headers=headers)
    print(response.images)
    if response.ok:
        target_image_path = os.path.join(UPLOAD_FOLDER, 'target_image.webp')
        with open(target_image_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Target image generated and saved to {target_image_path}")
        return target_image_path
    else:
        logging.error("Failed to fetch target image from Clipdrop")
        raise HTTPException(status_code=500, detail="Failed to fetch target image from Clipdrop")

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    logging.info("Starting face swap...")
    sourceImageFace = face_app.get(sourceImage)
    targetImageFace = face_app.get(targetImage)
    
    logging.info(f"Number of faces detected in source image: {len(sourceImageFace)}")
    logging.info(f"Number of faces detected in target image: {len(targetImageFace)}")

    if len(sourceImageFace) == 0 or len(targetImageFace) == 0:
        return None  # No faces detected
    
    sourceImageFaceSelected = sourceImageFace[0]
    targetImageFaceSelected = targetImageFace[0]

    img1_swapped = swapper.get(targetImage, targetImageFaceSelected,sourceImageFaceSelected, paste_back=True)
    
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
    
def add_two_logos_to_image(bg_img, logo_path1='1.png', logo_path2='2.png'):
    # Load the first logo (top-left)
    logo1 = cv2.imread(logo_path1, cv2.IMREAD_UNCHANGED)
    if logo1 is None:
        logging.error("First logo not found!")
        raise HTTPException(status_code=500, detail="First logo not found!")

    # Resize the first logo
    logo1 = cv2.resize(logo1, (50, 50))  # Adjust size as needed

    # Position the first logo at the top-left corner
    logo1_x = 50  # Position from the left edge
    logo1_y = 50  # Position from the top edge

    # Overlay the first logo on the background image
    if logo1.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_channel = logo1[:, :, 3] / 255.0
        for c in range(0, 3):
            bg_img[logo1_y:logo1_y + logo1.shape[0], logo1_x:logo1_x + logo1.shape[1], c] = (
                alpha_channel * logo1[:, :, c] +
                (1 - alpha_channel) * bg_img[logo1_y:logo1_y + logo1.shape[0], logo1_x:logo1_x + logo1.shape[1], c]
            )
    else:
        bg_img[logo1_y:logo1_y + logo1.shape[0], logo1_x:logo1_x + logo1.shape[1]] = logo1[:, :, :3]

    # Load the second logo (top-right)
    logo2 = cv2.imread(logo_path2, cv2.IMREAD_UNCHANGED)
    if logo2 is None:
        logging.error("Second logo not found!")
        raise HTTPException(status_code=500, detail="Second logo not found!")

    # Resize the second logo
    logo2 = cv2.resize(logo2, (100, 50))  # Adjust size as needed

    # Position the second logo at the top-right corner
    logo2_x = bg_img.shape[1] - logo2.shape[1] - 50  # Position from the right edge
    logo2_y = 50  # Position from the top edge (same as first logo)

    # Overlay the second logo on the background image
    if logo2.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_channel = logo2[:, :, 3] / 255.0
        for c in range(0, 3):
            bg_img[logo2_y:logo2_y + logo2.shape[0], logo2_x:logo2_x + logo2.shape[1], c] = (
                alpha_channel * logo2[:, :, c] +
                (1 - alpha_channel) * bg_img[logo2_y:logo2_y + logo2.shape[0], logo2_x:logo2_x + logo2.shape[1], c]
            )
    else:
        bg_img[logo2_y:logo2_y + logo2.shape[0], logo2_x:logo2_x + logo2.shape[1]] = logo2[:, :, :3]

    return bg_img

@app.post("/api/submit")
async def submit_user(user: User):
    try:
        # Insert user data into the 'users' table in Supabase
        response = supabase.table("users").insert({
            "name": user.name,
            "email": user.email
        }).execute()

        # Log the response from Supabase for debugging
        print(f"Supabase Response: {response.data}")  # Log the response data
        
        # Check if insertion was successful
     
        inserted_user = response.data[0]  # Get the inserted user data
        return {
                "id": inserted_user["id"],
                "name": inserted_user["name"],
                "email": inserted_user["email"]
            }
       
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error inserting user: {e}")



@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), prompt: str = Form("")):
    print(prompt)
    print(f"Received prompt: {prompt}")
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)

    sourceImage_cv = cv2.imread(img1_path)

    if sourceImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read source image with OpenCV: {img1_path}")

    logging.info(f"Source image shape: {sourceImage_cv.shape}")

    # Fetch target image from Clipdrop
    target_img_path = generate_target_imagePinokio(prompt)
    targetImage_cv = cv2.imread(target_img_path)

    if targetImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read target image from Clipdrop: {target_img_path}")

    logging.info(f"Target image shape: {targetImage_cv.shape}")

    swapped_image = simple_face_swap(sourceImage_cv, targetImage_cv, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    logging.info(f"Swapped image shape: {swapped_image.shape}")

    enhanced_image = enhance_face(swapped_image)

    logging.info(f"Enhanced image shape: {enhanced_image.shape}")

    #final_image = add_two_logos_to_image(enhanced_image, '1.png', '2.png') 
    final_image = enhanced_image

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, enhanced_image)

    logging.info(f"Image saved to: {result_path}")

    return FileResponse(result_path)

# HTTP server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)