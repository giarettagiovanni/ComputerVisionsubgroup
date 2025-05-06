import os
import sys
import time
import select
from datetime import datetime
import json
import cv2
from ultralytics import YOLO
#import supervision as sv
#import numpy as np

# Variables to modify (if needed)

MODEL_PATH = "/home/gianmarco/Documents/SensingRigs/CrabProject/Model/yolo11n.pt" # Path of YOLO model (absolute path)
CAMERA_ID = 0 # id of the camera (default is 0, if there's more than one camera change the integer)
INFERENCE_INTERVAL = 2 # interval (in seconds) for capturing data
SAVE_PATH = "captures" # leaf directory to save output images (will create a folder in path were this code is saved if it doesn't already exist)
SHOW_CAMERA_FEED = True # flag: True if you want to see the camera feed
CONF_THRESHOLD = 0.5 # threshold of confidence the inference must have in order to save
ESCAPE_KEY = 'q' # key to press in order to close the program

COLOR_BOX_AND_TEXT = (0,225,255) # color of bounding boxes and text
THICKNESS_BOX_AND_TEXT = 1 # thickness of lines and text
FONT_SIZE = 2 # scale the font size


# Process frame to make it gray-scale and blurred
def gray_and_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    return gray

# Checks if current and previous frame are different
def processed_movement_detection(prev_processed_frame,current_processed_frame):

    # Check if current and previous frame are different
    frame_diff = cv2.absdiff(prev_processed_frame, current_processed_frame)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Returns True if thresh is greater than the threshold
    return cv2.countNonZero(thresh) > 5000

# Annotate frame with bounding box, label, class and confidence
def annotate_frame(model, frame_to_annotate, box, color=(0, 225, 0), fontScale=1, thickness=2): # color and thickness are for both box and text
    # Extrapolate info from the model
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates

    # Draw box
    cv2.rectangle(frame_to_annotate, (x1, y1), (x2, y2), color, thickness)  # (x1,y1) is the top left corner, (x2,y2) the bottom right corner

    # Label with name, class, and confidence
    label = f"{class_name} {conf:.2f}"

    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontScale, thickness)

    # Calculate the position for the text
    text_x = x1 + 10
    text_y = y1 - 10

    # Check if the text goes beyond the left side of the frame
    if text_x < 0:
        text_x = 0

    # Check if the text goes beyond the top of the frame
    if text_y - text_height < 0:
        text_y = y1 + text_height + 10

    # Check if the text goes beyond the right side of the frame
    if text_x + text_width > frame_to_annotate.shape[1]:
        text_x = frame_to_annotate.shape[1] - text_width - 10

    # Check if the text goes beyond the bottom of the frame
    if text_y + text_height > frame_to_annotate.shape[0]:
        text_y = frame_to_annotate.shape[0] - text_height - 10

    # Draw the text
    cv2.putText(
        frame_to_annotate,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale,
        color,
        thickness,
        lineType=cv2.LINE_AA  # anti-aliased lines
    )

def append_to_log(model, box, detection_information):
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    xyxy = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    detection_information["detections"].append({
        "class": class_name,
        "confidence": round(conf, 3),
        "box": {
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3]
        }
    })

if __name__ == "__main__":
    print("Code started")

    # Create path for saving images (if it doesn't already exist)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model
    model = YOLO(MODEL_PATH,task="detect")

    # Initialize the standard OpenCV capture
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise Exception("Error: Could not open camera")

    # Per il confronto dei frame (movimento)
    ret, prev_frame = cap.read() #ret tells wether the reading was successful, prev_frame contains the captured image
    prev_gray = gray_and_blur(prev_frame)

    # Timer
    last_frameCheck_time = 0

    #image_counter = 0 # For naming files

    # Lista per il log
    log_data = []

    # MAIN LOOP: every INFERENCE_INTERVAL checks if there was movement from frame of previous interval. If there was movement, inference on current frame

    print(f"Main loop started. Press '{ESCAPE_KEY}' to quit.")

    try:
        while True:

            current_time = time.time()  # float representing number of seconds since the Unix epoch (January 1, 1970, UTC). Easier for comparisons than datetime.now()

            if (current_time - last_frameCheck_time) > INFERENCE_INTERVAL:

                ret, frame = cap.read() #ret tells whether the reading was successful, frame contains the captured image
                if not ret:
                    raise Exception("Error: Can't receive frame.")

                # Process the current frame, detect movement and save in prev_gray the processed current frame (to avoid processing 2 times the same frame)
                current_gray = gray_and_blur(frame)
                movement_detected = processed_movement_detection(prev_gray, current_gray)
                prev_gray = current_gray.copy()

                if movement_detected:

                    # Inference
                    results = model(frame)[0] #calls the model and performs inference on frame, [0] because it returns a list of results (in this case made of only one item)
                    #annotated_frame = results.plot() #plots without accounting for threshold

                    # Create copy of original image to plot on
                    annotated_frame = frame.copy()

                    # Filter confidence > CONF_THRESHOLD
                    confident_boxes = [
                        box for box in results.boxes if box.conf[0] > CONF_THRESHOLD
                    ]

                    if confident_boxes:

                        # Draw only "confident" boxes
                        for box in confident_boxes:
                            annotate_frame(model, annotated_frame, box, COLOR_BOX_AND_TEXT, FONT_SIZE, THICKNESS_BOX_AND_TEXT)

                        # Save annotated frame
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #different format from time.time(), needed for putting human-readable timestamps
                        timestamp_save_format = timestamp.replace(":", "_")
                        filename = f"capture {timestamp_save_format}.jpg" # filename con timestamp
                        #filename = f"captures/capture_{image_counter}.jpg" # filename con counter delle immagini
                        cv2.imwrite(f"{SAVE_PATH}/{filename}", annotated_frame) # Save the annotated frame

                        # Set detection information dictionary for this frame
                        detection_info = {
                            "filename": filename,
                            "timestamp": timestamp,
                            "detections": []
                        }
                        # Append to JSON log all the "confident boxes"
                        for box in confident_boxes:
                            append_to_log(model, box, detection_info)
                        log_data.append(detection_info)

                        print(f"[INFO] Image saved as: {filename}")
                        #image_counter += 1

                # Reset timer for inference
                last_frameCheck_time = current_time

            # Show camera feed
            if SHOW_CAMERA_FEED:
                cv2.imshow("YOLOv11 Detection", frame)
                # Stop the loop by pressing the q key if the showing feed option is enabled (in case ke
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Pressed the 'q' button. Closing the program.")
                    break
            elif select.select([sys.stdin],[], [], 0)[0]: # Checks if the user gave an input during the cycle
                user_input=sys.stdin.readline().strip() # Reads the input line (removing black spaces and everything that is not the escape key)
                # Stop the loop if the user pressed the escape key
                if user_input == ESCAPE_KEY:
                    time.sleep(0.01)
                    print(f"Pressed the '{ESCAPE_KEY}' key. Closing the program.")
                    break
                    
        # Save JSON log
        with open(f"{SAVE_PATH}/detections_log.json", "w") as log_file:
            json.dump(log_data, log_file, indent=4)

    except Exception as e:
        print(e)

    # Close resources that were opened
    cap.release() # camera feed (OpenCV capture)
    if SHOW_CAMERA_FEED: #close camera feed windows (the ones used to display the images)
        cv2.destroyAllWindows()
