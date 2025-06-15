import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # or yolo11, best-suited model

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Connect to RTSP stream using FFmpeg backend for stability
rtsp_url = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # backend avoids RTSP artifacts :contentReference[oaicite:1]{index=1}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        continue

    # Run YOLO detection
    results = model.predict(source=frame, stream=True)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            # Apply OCR
            ocr_results = ocr.ocr(crop)
            text = " ".join([line[1][0] for line in ocr_results])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("YOLO + OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
