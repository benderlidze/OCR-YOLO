#!/usr/bin/env python3
"""
Simple RTSP Stream Test Script
Tests basic connectivity to the RTSP stream without AI models
"""

import cv2
import time

# RTSP Stream configuration
RTSP_URL = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"

def test_rtsp_connection():
    print(f"Testing RTSP connection to: {RTSP_URL}")
    print("Attempting to connect...")
    
    # Create VideoCapture object
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    # Set some basic properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    
    if not cap.isOpened():
        print("❌ Failed to connect to RTSP stream")
        print("Possible issues:")
        print("1. Network connectivity")
        print("2. Incorrect credentials")
        print("3. Camera is offline")
        print("4. Firewall blocking connection")
        print("5. RTSP URL format incorrect")
        return False
    
    print("✅ Successfully connected to RTSP stream!")
    
    # Get stream properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Stream Properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {int(width)}x{int(height)}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        print("\nStarting stream viewer...")
        print("Press 'q' to quit, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add overlay text
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "RTSP Stream Test", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show every 10th frame info
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames: {frame_count}, FPS: {current_fps:.1f}")
            
            # Display frame
            cv2.imshow("RTSP Stream Test", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                filename = f"rtsp_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        total_time = time.time() - start_time
        print(f"\nSession complete:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count/total_time:.1f}" if total_time > 0 else "  Average FPS: N/A")
        
        cap.release()
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    print("=== RTSP Stream Connection Test ===")
    test_rtsp_connection()
