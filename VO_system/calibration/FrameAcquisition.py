import subprocess
import threading
import time
import os

# Number of images to capture
num_images = 2

# Delay between captures (in seconds)
delay = 1

# Output directories
left_dir = "calib_images/left_images"
right_dir = "calib_images/right_images"

# Create directories if they don't exist
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

# Define the camera device IDs
left_camera = "0"  # Modify based on your camera setup
right_camera = "1" # Modify based on your camera setup

def capture_image(camera, filename):
    """
    Capture a single image using libcamera-still.
    """
    command = [
        "libcamera-still",
        "--camera", camera,
	"-o", filename,
        "--fullscreen",
	"-t", "10000",
	"--width", "1920",
	"--height", "1080"
          # 1ms exposure time just to capture immediately
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Captured {filename}")
    except subprocess.CalledProcessError:
        print(f"Failed to capture {filename}")

time.sleep(2)
for i in range(0, 10):
    # Generate filenames
    left_filename = os.path.join(left_dir, f"left_{i:03d}.jpg")
    right_filename = os.path.join(right_dir, f"right_{i:03d}.jpg")
    
    # Create threads for parallel capture
    left_thread = threading.Thread(target=capture_image, args=(left_camera, left_filename))
    right_thread = threading.Thread(target=capture_image, args=(right_camera, right_filename))
    
    # Start threads
    left_thread.start()
    right_thread.start()
    
    # Wait for both to finish
    left_thread.join()
    right_thread.join()
    
    # Delay between captures
    time.sleep(delay)
