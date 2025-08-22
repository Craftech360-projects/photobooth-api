# Photobooth API Startup Instructions

This document provides step-by-step instructions for starting the Photobooth API backend on both Mac and Windows systems.

## Prerequisites

- Python 3.10+ installed on your system
- pip (Python package installer)
- At least 4GB of free disk space (for model downloads)
- Internet connection (for initial model downloads)

## Mac Setup

### 1. Open Terminal
Navigate to the project directory:
```bash
cd /path/to/photobooth-api
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install fastapi==0.115.5 uvicorn==0.32.0 python-multipart==0.0.17 insightface==0.7.3 opencv-python==4.10.0.84 gfpgan==1.3.8 numpy==1.26.4 pillow==10.4.0 requests==2.32.3 onnxruntime==1.22.1 torch==2.0.1 torchvision==0.15.2
```

### 5. Start the Server
```bash
python app.py
```

The server will start on `http://localhost:8000`

**Note:** On first startup, the server will automatically download required AI models (~200MB total). This may take a few minutes.

---

## Windows Setup

### 1. Open Command Prompt or PowerShell
Navigate to the project directory:
```cmd
cd C:\path\to\photobooth-api
```

### 2. Create Virtual Environment
```cmd
python -m venv env
```

### 3. Activate Virtual Environment
```cmd
env\Scripts\activate
```

### 4. Install Dependencies
```cmd
pip install fastapi==0.115.5 uvicorn==0.32.0 python-multipart==0.0.17 insightface==0.7.3 opencv-python==4.10.0.84 gfpgan==1.3.8 numpy==1.26.4 pillow==10.4.0 requests==2.32.3 onnxruntime==1.22.1 torch==2.0.1 torchvision==0.15.2
```

### 5. Start the Server
```cmd
python app.py
```

**Alternative:** Double-click the `start.bat` file (if virtual environment is already set up)

The server will start on `http://localhost:8000`

---

## API Documentation

Once the server is running, you can access:
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoint

- **POST** `/api/swap-face/`
  - Upload source and target images
  - Provide user name and email
  - Returns processed image with face swap

## Troubleshooting

### Common Issues:

1. **Port 8000 already in use**
   - Mac: `lsof -ti:8000 | xargs kill -9`
   - Windows: Find and kill the process using port 8000 in Task Manager

2. **Module not found errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements_clean.txt`

3. **Model download failures**
   - Check internet connection
   - Restart the server to retry downloads

4. **Memory issues**
   - Ensure at least 8GB RAM available
   - Close other applications to free memory

### Mac-specific Notes:
- Ensure Xcode command line tools are installed: `xcode-select --install`
- If using Apple Silicon (M1/M2), the models will run on CPU

### Windows-specific Notes:
- Ensure Microsoft Visual C++ Redistributable is installed
- Consider using Windows Subsystem for Linux (WSL) for better compatibility

## Stopping the Server

- Press `Ctrl+C` in the terminal where the server is running
- Or close the terminal/command prompt window

## Production Notes

For production deployment:
- Use `gunicorn` or similar WSGI server instead of the built-in uvicorn server
- Set up proper environment variables for configuration
- Consider using Docker for consistent deployment across environments