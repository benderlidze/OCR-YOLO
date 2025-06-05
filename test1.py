import cv2
from ultralytics import YOLO
import os
from fast_plate_ocr import ONNXPlateRecognizer
import csv

# Load YOLOv8 Nano model (pretrained or custom trained on license plates)
model = YOLO("license_plate_detector.pt")  # Replace with your custom model if available

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

# Open video stream (0 = default webcam, or provide path to video file)
cap = cv2.VideoCapture("20250605-1135-195857578_JpNYpqrH.mp4")  # Change to a video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Only process every 5th frame for speed optimization
    if frame_count % process_every_n_frames == 0:
        # Run YOLOv8 inference
        results = model(frame)

        # Extract and save detected plates
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size > 0:                # Save the plate image
                plate_path = f"plates/plate_{plate_count}.jpg"
                cv2.imwrite(plate_path, plate_img)                # Perform OCR using fast-plate-ocr
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
                
                # Print the detected license plate number
                if plate_text:
                    print(f"Frame {frame_count} - Detected plate #{plate_count}: {plate_text} (Confidence: {best_confidence:.3f})")
                else:
                    print(f"Frame {frame_count} - Plate #{plate_count}: No text detected")
                
                plate_count += 1

        # Visualize results
        annotated_frame = results[0].plot()
    else:
        # For frames we don't process, just show the original frame
        annotated_frame = frame

    # Display
    cv2.imshow("YOLOv8 License Plate Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
