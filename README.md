# Chat Assistant - phi-3 - ONNX

## Demo

(![demo gif](https://github.com/jivaniyash/Chat-Assistant-phi-3-ONNX/blob/master/demo-file/video-demo.gif))

## Usage
This repository provides a chat bot assistant using the phi-3 model powered by ONNX. Follow the instructions below to set up and run the application.

This app is build on top of [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/chat_app)

## Requirements
- Python 3.7 or higher
- virtualenv
- git
- git-lfs

## Steps to Start the App

### 1. Clone the Repository
```sh
git clone https://github.com/jivaniyash/Chat-Assistant-phi-3-ONNX
```

### 2. Set up a Virtual Environment
```sh
virtualenv venv
source venv/bin/activate
```

### 3. Install Git LFS & Download Model
```sh
sudo apt-get install git-lfs
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
```
This will download every file. If you want to run this inference on CPU devices, run the following commands:
```sh
mkdir models # Make directory
mv Phi-3-mini-4k-instruct-onnx/cpu_and_mobile models/cpu_and_mobile # Move the cpu_and_mobile directory into models directory
```
### 4. Install Python Requirements
```sh
pip install -r requirements.txt
```

### 5. Run the App
```sh
python ./chat_app/app.py
```



