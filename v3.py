import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from PIL import Image
import easyocr
import numpy as np
import re
import sqlite3
from datetime import datetime, timedelta
import time
from fast_plate_ocr import ONNXPlateRecognizer
import csv

# Initialize fast-plate-ocr recognizer
m = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')  # Load the ONNX model for OCR

# Create output directory for plates
os.makedirs("plates", exist_ok=True)
plate_count = 0

# Initialize CSV file for results
csv_filename = "detection_results.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Plate_ID', 'Detected_Text', 'Confidence_Score', 'Image_Path'])

# Frame processing control
frame_count = 0
process_every_n_frames = 5  # Process every 5th frame for speed


# RTSP Camera Information
rtsp_url = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"

# Connect to SQLite database
db_file = 'license_plates.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT,
    entry_time TEXT,
    exit_time TEXT,
    UNIQUE(plate, entry_time)
)
''')
conn.commit()

# Load the model
model_path = 'best.pt'
model = YOLO(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['tr'])

# Image processing function
def process_image(img):
    global plate_count, frame_count
    
    frame_count += 1
    
    # Only process every 5th frame for speed optimization
    if frame_count % process_every_n_frames == 0:
        # Make predictions using the model
        results = model(img)
        
        # Extract and save detected plates
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = img[y1:y2, x1:x2]
            if plate_img.size > 0:
                # Save the plate image
                plate_path = f"plates/plate_{plate_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
                cv2.imwrite(plate_path, plate_img)
                
                # Perform OCR using fast-plate-ocr
                try:
                    # Use return_confidence=True to get real confidence scores
                    ocr_result = m.run(plate_path, return_confidence=True)
                    if ocr_result and len(ocr_result) == 2 and len(ocr_result[0]) > 0:
                        plate_text = ocr_result[0][0]  # First detected text
                        confidence_scores = ocr_result[1][0]  # Confidence scores for each character
                        # Calculate average confidence across all characters
                        best_confidence = float(confidence_scores.mean())
                    else:
                        plate_text = ""
                        best_confidence = 0.0
                except Exception as e:
                    print(f"OCR error for plate {plate_count}: {e}")
                    plate_text = ""
                    best_confidence = 0.0
                
                # Save results to CSV
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        f"plate_{plate_count}",
                        plate_text if plate_text else "No text detected",
                        f"{best_confidence:.3f}",
                        plate_path
                    ])
                
                # Draw the detected box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add detected text to the image
                if plate_text:
                    display_text = f"{plate_text} ({best_confidence:.3f})"
                    cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    print(f"Frame {frame_count} - Detected plate #{plate_count}: {plate_text} (Confidence: {best_confidence:.3f})")
                else:
                    cv2.putText(img, "Plate Not Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    print(f"Frame {frame_count} - Plate #{plate_count}: No text detected")
                
                plate_count += 1
        
        # Visualize results
        annotated_frame = results[0].plot()
        return annotated_frame
    else:
        # For frames we don't process, just return the original frame
        return img

# Video processing function
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        processed_frame = process_image(frame)
        out.write(processed_frame)

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")

def main(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_path.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(input_path)
        output_path = os.path.join(output_dir, 'result.jpg')
        processed_img = process_image(img)
        cv2.imwrite(output_path, processed_img)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.show()
    elif input_path.endswith(('.mp4', '.avi', '.mov')):
        output_path = os.path.join(output_dir, 'result_video.mp4')
        process_video(input_path, output_path)

    elif input_path == 'camera':
        # Capture from camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            processed_frame = process_image(frame)

            # Show the processed frame
            cv2.imshow('Live Feed', processed_frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif input_path.startswith('rtsp://'):
        # Capture from camera stream
        while True:
            try:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise ValueError("Camera connection failed.")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError("Frame could not be read, stream ended.")
                    
                    processed_frame = process_image(frame)
                    cv2.imshow('Stream', processed_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                break  # Break loop to restart

            except Exception as e:
                print(f"An error occurred: {e}")
                cap.release()
                cv2.destroyAllWindows()
                print("Waiting 5 seconds and reconnecting...")
                time.sleep(5)  # Wait 5 seconds and retry

    else:
        print("Unsupported input type.")

# Example usage for external camera stream
input_path = rtsp_url
output_dir = 'output_directory'
main(input_path, output_dir)