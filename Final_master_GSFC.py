import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *

# Load your custom YOLO model
#model = YOLO("/home/dnw2/Desktop/Sugosa/sugosa/sugosa_clean.pt")
model = YOLO("/home/nasscom-gh-nwarch-ai/Documents/Siddharth/GSFC/GSFC_SIDD/inferencing/sugosa_refrence_gsfc/GSFC_11_5_24.pt")

# Open the video file
video_path = "/home/nasscom-gh-nwarch-ai/Documents/Siddharth/GSFC/GSFC_SIDD/inferencing/sugosa_refrence_gsfc/01.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_path = '/home/dnw2/Desktop/Sugosa/Runway_test/47_test1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

# Initialize SORT tracker
tracker = Sort(max_age=10, min_hits=0, iou_threshold=0.3)

# Variables for the line ROI
line_points = []  # List to store the two points of the line
line_set = False  # Flag to indicate if the line has been set

# Counters for gunny bags
bags_in_count = 0
bags_out_count = 0

# Set to keep track of counted object IDs for in and out directions
counted_in = set()
counted_out = set()

# To store previous centers of objects for line-crossing detection
previous_centers = {}

def draw_line(event, x, y, flags, param):
    global line_points, line_set
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"Point {len(line_points)} set at: {x}, {y}")
        if len(line_points) == 2:
            line_set = True
            print("Line set. Starting inference...")

# Function to determine if an object has crossed the line
def is_crossing_line(center, previous_center, line_start, line_end):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    # Get signs for both the current and previous centers relative to the line
    d1 = sign(center, line_start, line_end)
    d2 = sign(previous_center, line_start, line_end)
    
    # If the signs differ, the object has crossed the line
    return d1 * d2 < 0

# Function to check the crossing direction (in or out)
def get_crossing_direction(center, previous_center):
    if previous_center[1] > center[1]:  # Moving upward (bag coming in)
        return "in"
    elif previous_center[1] < center[1]:  # Moving downwards (bag going out)
        return "out"
    return None

# Set mouse callback function
cv2.namedWindow('Select Line Points')
cv2.setMouseCallback('Select Line Points', draw_line)

# Read the first frame to set up the line
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Resize the first frame for line selection
resized_frame = cv2.resize(frame, (1280, 720))

# Display the first frame for line selection
while not line_set:
    cv2.imshow('Select Line Points', resized_frame)  # Show the resized frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('Select Line Points')  # Close the selection window

# Scale the line points back to the original frame size
if len(line_points) == 2:
    scale_x = original_width / 1280
    scale_y = original_height / 720
    line_points = [(int(x * scale_x), int(y * scale_y)) for (x, y) in line_points]

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model to detect objects in the frame
    results = model(frame)

    # Extract bounding boxes, confidence scores, and class labels from YOLO output
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    scores = results[0].boxes.conf.cpu().numpy()      # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()      # Class labels

    # Prepare detections for SORT tracker
    detections_with_scores = np.hstack((detections, scores.reshape(-1, 1)))

    # Update tracker with the current frame detections
    tracked_objects = tracker.update(detections_with_scores)

    # If the line is set, process objects crossing the line
    if line_set:
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Check if the object has crossed the line
            if obj_id in previous_centers:
                if is_crossing_line(center, previous_centers[obj_id], line_points[0], line_points[1]):
                    direction = get_crossing_direction(center, previous_centers[obj_id])

                    # Count based on the direction and ensure that each bag is counted only once for each direction
                    if direction == "in" and obj_id not in counted_in:
                        bags_in_count += 1
                        counted_in.add(obj_id)
                    elif direction == "out" and obj_id not in counted_out:
                        bags_out_count += 1
                        counted_out.add(obj_id)

            # Store the current center as the previous center for the next frame
            previous_centers[obj_id] = center

            # Draw bounding boxes and track IDs on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw the line if set (in the original frame dimensions)
    if len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

        # Calculate total bags in (bags_in - bags_out)
        total_bags_in = bags_in_count - bags_out_count

        # Display separate counts for bags moving "in", "out", and the total bags currently in
        cv2.putText(frame, f"Bag In: {bags_in_count}", (42, 318), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bag Out: {bags_out_count}", (43, 368), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Bags In: {total_bags_in}", (44, 415), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    # Resize the frame to match output dimensions before saving
    resized_frame = cv2.resize(frame, (1280, 720))

    # Write the resized frame with tracking results to the output video
    out.write(resized_frame)

    # Show the frame in the window
    cv2.imshow('Tracking', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracked video saved at {output_path}")
print(f"Total gunny bags counted in: {bags_in_count}")
print(f"Total gunny bags counted out: {bags_out_count}")
print(f"Current total bags in: {bags_in_count - bags_out_count}")
