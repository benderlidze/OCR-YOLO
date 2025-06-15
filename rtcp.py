import cv2

# Define the RTSP URL
RTSP_URL = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"

# Open the RTSP stream
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Check if the stream is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open RTSP stream: {RTSP_URL}")
    exit(1)

print(f"Successfully connected to RTSP stream: {RTSP_URL}")

# Read and display frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from RTSP stream")
        break

    # Display the frame
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()