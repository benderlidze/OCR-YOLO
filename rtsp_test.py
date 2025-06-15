import vlc
import time
import cv2
import numpy as np

RTSP_URL = "rtsp://admin:Rade13245@213.5.194.11:100/Streaming/Channels/101"

def stream_with_vlc():
    """Stream RTSP using VLC backend"""
    
    # Create VLC instance with network options
    vlc_args = [
        '--intf', 'dummy',
        '--no-audio',
        '--network-caching=300',
        '--rtsp-tcp',  # Force TCP for RTSP
        '--live-caching=300'
    ]
    
    instance = vlc.Instance(vlc_args)
    player = instance.media_player_new()
    
    # Create media
    media = instance.media_new(RTSP_URL)
    player.set_media(media)
    
    print("Starting VLC RTSP stream...")
    player.play()
    
    # Wait for stream to start
    time.sleep(3)
    
    try:
        while True:
            state = player.get_state()
            
            if state == vlc.State.Playing:
                print("✓ Stream is playing")
                break
            elif state == vlc.State.Error:
                print("✗ VLC error occurred")
                return
            elif state == vlc.State.Ended:
                print("Stream ended")
                return
            
            time.sleep(0.5)
        
        # Keep alive
        input("Stream is running. Press Enter to stop...")
        
    except KeyboardInterrupt:
        print("Interrupted")
    
    finally:
        player.stop()
        print("VLC stream stopped")

if __name__ == "__main__":
    stream_with_vlc()