1) Enable Long Path Support

Installing collected packages: onnx, mdurl, MarkupSafe, imageio-ffmpeg, idna, grpcio, future, fonttools, ffmpy, decorator, cv, beautifulsoup4, aiofiles, uvicorn, proglog, markdown-it-py, torchvision, tb-nightly, starlette, rich, moviepy, flask, typer, timm, insightface, gradio-client, gdown, fastapi, facexlib, basicsr, gradio, gfpgan
ERROR: Could not install packages due to an OSError: [WinError 206] The filename or extension is too long: 'C:\\Users\\dhrit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages\\onnx\\backend\\test\\data\\node\\test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False\\test_data_set_0'


[notice] A new release of pip is available: 24.3.1 -> 25.0
[notice] To update, run: C:\Users\dhrit\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip

 SOLUTION: Press Win + R, type regedit, and press Enter.
    Navigate to:
    HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    Double-click the entry LongPathsEnabled and set its value to 1.
    Restart your system.
