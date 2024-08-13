import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import yolov5
import time
import easyocr

def load_model(model_path):
    """Load the YOLOv8 model for vehicle detection and tracking."""
    return YOLO(model_path)

def load_license_plate_model(model_path):
    """Load the YOLOv5 model for license plate detection."""
    model = yolov5.load(model_path)
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model

def initialize_video(video_path):
    """Initialize video capture from the given path."""
    return cv2.VideoCapture(video_path)
    #return cv2.VideoCapture(0)

def initialize_trackers():
    """Initialize ByteTrack and annotators."""
    byte_tracker = sv.ByteTrack()
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(thickness=1)
    return byte_tracker, bounding_box_annotator, label_annotator, trace_annotator

def process_frame(model, frame, byte_tracker):
    """Process a single frame for detections and tracking."""
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    return detections

def detect_license_plate(license_plate_model, frame):
    """Detect license plates in the frame."""
    results = license_plate_model(frame, size=640)
    predictions = results.pred[0]
    return predictions

def extract_license_plate_text(license_plate_img, reader):
    """Extract text from the license plate image using EasyOCR."""
    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    if result:
        return result[0][-2]
    return "NULL"

def create_labels(detections, detection_id, license_plate_data, timestamp):
    """Create labels for detected objects and update the detection ID."""
    labels = []
    detection_entries = []
    for i in range(len(detections)):
        detection_id += 1
        class_name = detections['class_name'][i]
        confidence = detections.confidence[i]
        bbox = detections.xyxy[i]
        license_plate_coord = license_plate_data[i]['coords'] if i < len(license_plate_data) else None
        license_plate_text = license_plate_data[i]['text'] if i < len(license_plate_data) else "NULL"
        labels.append(f"ID {detection_id}: {class_name} {confidence:.2f}")
        detection_entries.append({
            "Timestamp": timestamp,
            "Detection ID": detection_id,
            "Class": class_name,
            "Confidence": confidence,
            "Bounding Box": bbox.tolist(),
            "License Plate Bounding Box": license_plate_coord.tolist() if license_plate_coord is not None else "NULL",
            "License Plate Text": license_plate_text
        })
    return labels, detection_entries, detection_id

def annotate_frame(frame, detections, labels, bounding_box_annotator, label_annotator, trace_annotator):
    """Annotate the frame with bounding boxes, labels, and traces."""
    frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = trace_annotator.annotate(scene=frame, detections=detections)
    return frame

def main():
    model_path = "yolov8x.pt"
    license_plate_model_path = "keremberke/yolov5m-license-plate"
    video_path = "C:\\Universal Folder\\PS\\Untitled design (2).mp4"
    output_excel = "detections.xlsx"

    # Load models and initialize video
    model = load_model(model_path)
    license_plate_model = load_license_plate_model(license_plate_model_path)
    cap = initialize_video(video_path)

    # Initialize trackers and annotators
    byte_tracker, bounding_box_annotator, label_annotator, trace_annotator = initialize_trackers()

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])

    # DataFrame to store detections
    detections_df = pd.DataFrame(columns=["Timestamp", "Detection ID", "Class", "Confidence", "Bounding Box", "License Plate Bounding Box", "License Plate Text"])
    detection_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Process the frame
        detections = process_frame(model, frame, byte_tracker)

        # Detect license plates within the detected vehicles
        license_plate_data = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_roi = frame[y1:y2, x1:x2]
            license_plate_predictions = detect_license_plate(license_plate_model, vehicle_roi)
            if len(license_plate_predictions) > 0:
                lp_coords = license_plate_predictions[0][:4]
                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_coords)
                license_plate_img = vehicle_roi[lp_y1:lp_y2, lp_x1:lp_x2]
                license_plate_text = extract_license_plate_text(license_plate_img, reader)
                license_plate_data.append({"coords": lp_coords, "text": license_plate_text})
            else:
                license_plate_data.append({"coords": None, "text": "NULL"})

        # Create labels and update detection entries
        labels, detection_entries, detection_id = create_labels(detections, detection_id, license_plate_data, timestamp)

        # Update the DataFrame with new detections
        for entry in detection_entries:
            detections_df = detections_df._append(entry, ignore_index=True)

        # Annotate and display the frame
        annotated_frame = annotate_frame(frame, detections, labels, bounding_box_annotator, label_annotator, trace_annotator)
        cv2.imshow("YOLOv8 Vehicle and License Plate Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Save detections to Excel
    detections_df.to_excel(output_excel, index=False)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed and displayed all frames. Detections saved to {output_excel}.")

if __name__ == "__main__":
    main()
