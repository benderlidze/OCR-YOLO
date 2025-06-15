import cv2
from open_image_models import LicensePlateDetector
import os
from fast_plate_ocr import ONNXPlateRecognizer
import csv

# Load YOLOv9-t-384-license-plate-end2end model from open-image-models
print("Initializing YOLOv9-t-384-license-plate-end2end detector...")
model = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

# Initialize fast-plate-ocr recognizer
m = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')  # Load the ONNX model for OCR

# Create output directory for plates
os.makedirs("plates", exist_ok=True)
plate_count = 0

# Initialize CSV file for results
csv_filename = "detection_results.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Plate_ID', 'Frame_Number', 'Detected_Text', 'OCR_Confidence', 'Detection_Confidence', 'Image_Path'])

# Frame processing control
frame_count = 0
process_every_n_frames = 2  # Process every 5th frame for speed

# Open video stream (0 = default webcam, or provide path to video file)
cap = cv2.VideoCapture("20250605-1135-195857578_JpNYpqrH.mp4")  # Change to a video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
      # Only process every 5th frame for speed optimization
    if frame_count % process_every_n_frames == 0:
        # Run YOLOv9 inference using open-image-models
        detections = model.predict(frame)        # Extract and save detected plates
        if detections is not None and len(detections) > 0:
            for detection in detections:
                # Extract bounding box coordinates from detection object
                # DetectionResult has bounding_box attribute with x1, y1, x2, y2 properties
                bbox = detection.bounding_box
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                
                # Extract plate region from frame
                plate_img = frame[y1:y2, x1:x2]
                
                if plate_img.size > 0:
                    # Save the plate image
                    plate_path = f"plates/plate_{plate_count}.jpg"
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
                    
                    # Print the detected license plate number with detection confidence
                    detection_conf = detection.confidence if hasattr(detection, 'confidence') else 0.0
                    if plate_text:
                        print(f"Frame {frame_count} - Detected plate #{plate_count}: {plate_text} (OCR: {best_confidence:.3f}, Detection: {detection_conf:.3f})")
                    else:
                        print(f"Frame {frame_count} - Plate #{plate_count}: No text detected (Detection: {detection_conf:.3f})")
                    
                    plate_count += 1

        # Visualize results using open-image-models display method
        if detections is not None and len(detections) > 0:
            annotated_frame = model.display_predictions(frame)
        else:
            annotated_frame = frame
    else:
        # For frames we don't process, just show the original frame
        annotated_frame = frame    # Display
    cv2.imshow("YOLOv9 License Plate Detection (Open Image Models)", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
