In this FastAPI application, the /api/swap-face/ endpoint is returning an enhanced image file to the client in JPEG format (saved with the .jpg extension). This is achieved using the FileResponse class from FastAPI, which sends the image file directly to the client.

Here is a breakdown of the process:

File Upload:
The client uploads two images (sourceImage and targetImage).

Processing Pipeline:

1. The images are loaded using OpenCV.
2. Face swapping is performed using single_face_swap (or two_face_swap, if enabled).
3. The swapped image is enhanced using the GFPGAN model (enhance_face).
 Result Saving:
4. The final enhanced image is saved as a JPEG file in the RESULT_FOLDER.

Response:
5. The FileResponse sends the enhanced image back to the client, allowing them to download the processed image file.