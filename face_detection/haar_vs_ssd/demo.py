import cv2
import numpy as np
import pandas as pd
import time

# Load the Haar cascade xml file for face detection.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load the SSD model.
ssd_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
)

# Open a handle to the webcam.
video_capture = cv2.VideoCapture(0)

frame_count = 0
total_time = 0

# Initialize a list to store the analytics data
analytics_data_haar = []
analytics_data_ssd = []

while True:
    # Calculate FPS
    start_time = time.time()

    # Read a frame from the webcam.
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image properties
    height, width, channels = frame.shape

    # Detect faces using Haar cascades
    model_start_time = time.time()
    faces_haar = face_cascade.detectMultiScale(gray, 1.3, 5)
    model_end_time = time.time()
    model_time_haar = model_end_time - model_start_time

    # Resize the frame to 300x300 for SSD
    image = cv2.resize(frame, (300, 300))
    aspect_ratio_x = frame.shape[1] / 300
    aspect_ratio_y = frame.shape[0] / 300

    # Pre-process the image for the SSD model
    imageBlob = cv2.dnn.blobFromImage(image)

    # Detect faces using SSD
    model_start_time = time.time()
    ssd_detector.setInput(imageBlob)
    detections_ssd = ssd_detector.forward()
    model_end_time = time.time()
    model_time_ssd = model_end_time - model_start_time

    # Number of faces and average confidence
    num_faces_haar = len(faces_haar)
    num_faces_ssd = len(detections_ssd[0])
    avg_confidence_ssd = np.mean([d[2] for d in detections_ssd[0][0]])

    # Calculate FPS and append data to DataFrame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    total_time += end_time - start_time
    avg_fps = frame_count / total_time

    # Append the analytics data to the lists
    analytics_data_haar.append(
        [frame_count, avg_fps, model_time_haar, num_faces_haar, width, height, channels]
    )
    analytics_data_ssd.append(
        [
            frame_count,
            avg_fps,
            model_time_ssd,
            num_faces_ssd,
            width,
            height,
            channels,
            avg_confidence_ssd,
        ]
    )

    frame_count += 1

    # Loop over the faces detected by Haar
    for x, y, w, h in faces_haar:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Haar", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Format the detections into a DataFrame
    column_labels = [
        "img_id",
        "is_face",
        "confidence",
        "left",
        "top",
        "right",
        "bottom",
    ]
    detections_df = pd.DataFrame(detections_ssd[0][0], columns=column_labels)

    # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
    detections_df = detections_df[detections_df["is_face"] == 1]
    detections_df = detections_df[detections_df["confidence"] >= 0.90]

    # Loop over the faces detected by SSD
    for i, instance in detections_df.iterrows():
        confidence_score = round(100 * instance["confidence"], 2)
        left = instance["left"] * 300
        bottom = instance["bottom"] * 300
        right = instance["right"] * 300
        top = instance["top"] * 300

        # Draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence_score)
        y = top * aspect_ratio_y - 10 if top > 20 else top * aspect_ratio_y + 10
        cv2.rectangle(
            frame,
            (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),
            (int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "SSD: " + text,
            (int(left * aspect_ratio_x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

    # Show the output frame
    cv2.imshow("Face detection with Haar and SSD", frame)

    # Break from the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Convert the analytics data to a DataFrame and save to csv files
column_labels_haar = [
    "frame_count",
    "fps",
    "model_time",
    "num_faces",
    "image_width",
    "image_height",
    "image_channels",
]
analytics_df_haar = pd.DataFrame(analytics_data_haar, columns=column_labels_haar)
analytics_df_haar.to_csv("analytics_haar.csv", index=False)

column_labels_ssd = [
    "frame_count",
    "fps",
    "model_time",
    "num_faces",
    "image_width",
    "image_height",
    "image_channels",
    "avg_confidence",
]
analytics_df_ssd = pd.DataFrame(analytics_data_ssd, columns=column_labels_ssd)
analytics_df_ssd.to_csv("analytics_ssd.csv", index=False)

# Open the report file in write mode
with open("report.txt", "w") as f:
    # Write the report to the file
    f.write("=== Summary Report for Haar ===\n")
    f.write(f"Total frames processed: {frame_count}\n")
    f.write(f"Average FPS: {analytics_df_haar['fps'].mean()}\n")
    f.write(f"Total model time: {analytics_df_haar['model_time'].sum()} seconds\n")
    f.write(
        f"Average model time per frame: {analytics_df_haar['model_time'].mean()} seconds\n"
    )
    f.write(f"Total number of faces detected: {analytics_df_haar['num_faces'].sum()}\n")
    f.write(
        f"Average number of faces per frame: {analytics_df_haar['num_faces'].mean()} faces\n"
    )
    f.write(f"Average image width: {analytics_df_haar['image_width'].mean()}\n")
    f.write(f"Average image height: {analytics_df_haar['image_height'].mean()}\n")
    f.write(f"Average image channels: {analytics_df_haar['image_channels'].mean()}\n")

    f.write("\n=== Summary Report for SSD ===\n")
    f.write(f"Total frames processed: {frame_count}\n")
    f.write(f"Average FPS: {analytics_df_ssd['fps'].mean()}\n")
    f.write(f"Total model time: {analytics_df_ssd['model_time'].sum()} seconds\n")
    f.write(
        f"Average model time per frame: {analytics_df_ssd['model_time'].mean()} seconds\n"
    )
    f.write(f"Total number of faces detected: {analytics_df_ssd['num_faces'].sum()}\n")
    f.write(
        f"Average number of faces per frame: {analytics_df_ssd['num_faces'].mean()} faces\n"
    )
    f.write(f"Average image width: {analytics_df_ssd['image_width'].mean()}\n")
    f.write(f"Average image height: {analytics_df_ssd['image_height'].mean()}\n")
    f.write(f"Average image channels: {analytics_df_ssd['image_channels'].mean()}\n")
    f.write(f"Average confidence: {analytics_df_ssd['avg_confidence'].mean()}\n")
