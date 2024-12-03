from ultralytics import YOLO
import cv2
import numpy as np

# Load the models
person_truck_model = YOLO('/home/nasscom-gh-nwarch-ai/Documents/Siddharth/Saint_Gobian/person_truck_16_10_24.pt')  # Person & Truck Detection Model
harness_model = YOLO('/home/nasscom-gh-nwarch-ai/Documents/Siddharth/Saint_Gobian/saint_gobian_harness_24_10_24.pt')  # Harness Segmentation Model

def detect_objects(image, model, classes):
    results = model.predict(image, classes=classes, verbose=False, imgsz=[1280], conf=0.55, device=0)[0]
    return results

def get_bottom_coordinate(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return x1, y1, x2, y2

def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def is_person_inside_truck(person_bbox, truck_bbox):
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    truck_x1, truck_y1, truck_x2, truck_y2 = truck_bbox
    return (truck_x1 <= person_x1 <= truck_x2 or truck_x1 <= person_x2 <= truck_x2) and (truck_y1 <= person_y2 <= 0.95 * truck_y2)

def draw_polygon(mask, frame, color=(0, 0, 255), thickness=15):  # Increased thickness to 15
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=thickness)

def process_frame(frame):
    frame_resized = cv2.resize(frame, (640, 360))

    # Detect person and truck
    detections = detect_objects(frame_resized, person_truck_model, classes=[0, 1])
    person_bboxes = []
    truck_bboxes = []

    # Collect all bounding boxes for persons and trucks
    for i, bbox in enumerate(detections.boxes.xyxy):
        class_id = int(detections.boxes.cls[i])
        if class_id == 0:  # Assuming 0 is for Person
            person_bboxes.append(get_bottom_coordinate(bbox.cpu().numpy()))
        elif class_id == 1:  # Assuming 1 is for Truck
            truck_bboxes.append(get_bottom_coordinate(bbox.cpu().numpy()))

    # Find the truck with the maximum area
    if truck_bboxes:
        truck_bboxes = [max(truck_bboxes, key=calculate_area)]  # Keep only the truck with the maximum area

    # Iterate over each person and check if they are inside the truck
    for truck_bbox in truck_bboxes:
        persons_inside_truck = []  # To store persons inside the truck

        for person_bbox in person_bboxes:
            if is_person_inside_truck(person_bbox, truck_bbox):
                persons_inside_truck.append(person_bbox)
                # Detect harness using segmentation model only if person is inside the truck
                harness_results = detect_objects(frame_resized, harness_model, classes=[0])  # Assuming harness is class 0
                harness_detected = False

                if harness_results.masks:
                    harness_detected = True
                    for mask in harness_results.masks.cpu().numpy():
                        if isinstance(mask, np.ndarray) and mask.size > 0:  # Check if mask is valid
                            mask_resized = cv2.resize(mask, (640, 360))
                            # Draw harness with red color and increased thickness
                            draw_polygon(mask_resized, frame_resized, color=(0, 0, 255), thickness=15)
                        else:
                            print("Warning: Invalid mask detected.")

                # Check harness status and print result
                if harness_detected:
                    cv2.putText(frame_resized, "Harness Detected", (person_bbox[0], person_bbox[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print("Person is safe (wearing harness)")
                else:
                    cv2.putText(frame_resized, "No Harness", (person_bbox[0], person_bbox[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Person is unsafe (not wearing harness)")
        
        # Draw bounding boxes only for persons inside the truck
        for person_bbox in persons_inside_truck:
            cv2.rectangle(frame_resized, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 255, 0), 1)
            cv2.putText(frame_resized, "Person", (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw the bounding box for the truck with the maximum area
    for truck_bbox in truck_bboxes:
        cv2.rectangle(frame_resized, (truck_bbox[0], truck_bbox[1]), (truck_bbox[2], int(1.0 * truck_bbox[3])), (255, 0, 0), 1)
        cv2.putText(frame_resized, "Truck", (truck_bbox[0], truck_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame_resized

def main():
    cap = cv2.VideoCapture('/home/nasscom-gh-nwarch-ai/Documents/Siddharth/Saint_Gobian/26 (online-video-cutter.com).mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Frame', processed_frame)

        if cv2.waitKey(90) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
