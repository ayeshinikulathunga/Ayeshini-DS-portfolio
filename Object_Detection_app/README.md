# Object Detection Project with YOLOv8

This project provides a comprehensive, hands-on introduction to Object Detection using the powerful YOLOv8 (You Only Look Once, version 8) model from Ultralytics. 
It is organized as a single Google Colab notebook, divided into five distinct and executable parts.

## Libraries and Datasets used
1. Ultralytics (YOLOv8) - The main library for the object detection model
2. Opencv-python-headless (cv2) - handles all image and video processing tasks
3. matplotlib, pillow - used for displaying images and plots
4. streamlit, pyngrok - used to build and expose the web app
5. coco128 - A small, 128-image subset of the COCO dataset, used for quick model traning

## How to use
1. Go to the Object_Detection.ipnyb file and open it in Google Colab.
2. Run the cells sequentially from top to bottom
3. When you reach sections that require file input, a file upload widget will
   appear and you can test them with sample images and videos
4. For the "Live Webcam Detection" section, your browser will prompt you to
   allow webcam access.
5. For the final section (Part 5), you must replace the placeholder NGROK_AUTH_TOKEN with your personal
   Ngrok authentication token to launch the web app.

# Project Structure
Part 1: Image Detection
* Goal: Perform object detection on a single static image.
* Process: The script prompts you to upload an image (e.g., JPG, PNG). It then loads the pre-trained
  yolov8n.pt model, runs inference and displays the annotated image with detected objects and bounding boxes.

Part 2: Object Detection on Video
* Goal: Apply object detection across every frame of a video file.
* Process: You are prompted to upload a video file (e.g., MP4). The script processes the video frame-by-frame,
  annotates each one and saves the results to a new video file (output_detected.mp4). The finished,
  annotated video is then automatically downloaded.

Part 3 : Live Webcam Detection
* Goal: Capture a live image from your camera and detect objects in it.
* Process: A custom function initiates your device's webcam feed within the notebook.
  You click a "Capture" button, and the resulting snapshot is saved. YOLOv8 then runs
  inference on this captured image, and the annotated result is immediately displayed.

Part 4 : Mini YOLO Training
* Goal: Demonstrate how to fine-tune a YOLOv8 model on a custom dataset.
* Process: This section automatically downloads the small coco128 dataset.
  It then configures and trains the base yolov8n.pt model for a short period (10 epochs) using this dataset.
  This creates a custom model which is then used to predict objects on a sample image from the dataset.

Part 5 : Simple Streamlit App (Deployment)
* Goal: Deploy the object detection functionality into an interactive web application.
* Process: This is the final, comprehensive part. It writes all the detection logic into a Streamlit application file (app_streamlit.py).
  It then uses pyngrok to tunnel the local Streamlit service to a public URL.
  The resulting app allows users to:
    * Upload images or videos for detection
    * Customize detection parameters (e.g., confidence threshold)
    * Optionally upload their own custom trained YOLO weights
    * View the annotated results and a structured data table of detections
