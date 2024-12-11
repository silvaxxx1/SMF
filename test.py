from ultralytics import YOLO
import cv2

# Load the pre-trained model (best.pt is your trained model)
model = YOLO("models/best.pt")  # Replace with your trained model path

# Define the classes you're interested in
classes_of_interest = {
    0: "Hardhat",        # Hardhat
    1: "Mask",           # Mask
    2: "No Hardhat",     # No Hardhat
    3: "No Mask",        # No Mask
    4: "No Safety Vest", # No Safety Vest
    5: "Person",         # Person
    7: "Safety Vest"     # Safety Vest
}

# Open the video file
video_path = "is2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output video
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform inference on the frame (YOLO automatically resizes and normalizes the image)
    results = model(frame)  # Perform inference

    # Get the class IDs of the detections
    class_ids = results[0].boxes.cls.tolist()

    # Filter the results to include only the classes of interest
    filtered_results = []
    for i, class_id in enumerate(class_ids):
        if class_id in classes_of_interest:
            # Add the bounding box and the class label
            filtered_results.append((results[0].boxes.xyxy[i], classes_of_interest[class_id]))

    # Draw the bounding boxes for the filtered classes on the frame
    for box, class_name in filtered_results:
        x1, y1, x2, y2 = map(int, box)

        if class_name == "Person":
            # Draw a bounding box for Person with no text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        else:
            # For other classes, draw only text above the bounding box
            cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Optionally, display the frame with bounding boxes
    cv2.imshow("Processed Video", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
