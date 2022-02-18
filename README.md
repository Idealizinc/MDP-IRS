MDP-IRS

Setup Steps
1) Install CUDA 11.3 https://developer.nvidia.com/cuda-11.3.0-download-archive

2) cd to project root, 
2.1) pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
2.2) pip install -r yolov5/requirements.txt

3) Run the ImageRecognitionServer.py
3.1) cd mdp-irs\src
3.2) python ImageRecognitionServer.py