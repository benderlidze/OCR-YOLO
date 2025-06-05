import cv2

# RTSP stream URL
RTSP_URL = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"

def view_rtsp_stream():
    """Connects to an RTSP stream and displays it."""
    print(f"Attempting to connect to RTSP stream: {RTSP_URL}")

    # Use FFMPEG backend for better RTSP support
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        print("Please check the following:")
        print("1. The camera is powered on and connected to the network.")
        print("2. The RTSP URL, username, and password are correct.")
        print("3. Your firewall is not blocking the connection.")
        print("4. OpenCV is correctly installed with FFMPEG support.")
        return

    print("Successfully connected to RTSP stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from stream. Stream may have ended or there was a connection issue.")
            break

        # Display the resulting frame
        cv2.imshow('RTSP Stream Viewer', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting viewer...")
            break

    # When everything is done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == '__main__':
    view_rtsp_stream()

