import cv2
import numpy as np
import pandas as pd
import time

# Load the Haar cascade xml file for face detection.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open a handle to the webcam.
video_capture = cv2.VideoCapture(0)

frame_count = 0
total_time = 0

# Initialize a list to store the analytics data
analytics_data = []

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
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time

    # Number of faces
    num_faces = len(faces)

    # Calculate FPS and append data to DataFrame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    total_time += end_time - start_time
    avg_fps = frame_count / total_time

    # Append the analytics data to the list
    analytics_data.append(
        [frame_count, avg_fps, model_time, num_faces, width, height, channels]
    )

    frame_count += 1

    # Loop over the faces
    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Face detection with Haar cascades", frame)

    # Break from the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Convert the analytics data to a DataFrame and save to a csv file
column_labels = [
    "frame_count",
    "fps",
    "model_time",
    "num_faces",
    "image_width",
    "image_height",
    "image_channels",
]
analytics_df = pd.DataFrame(analytics_data, columns=column_labels)
analytics_df.to_csv("analytics.csv", index=False)

# Open the report file in write mode
with open("report.txt", "w") as f:
    # Write the report to the file
    f.write("=== Summary Report ===\n")
    f.write(f"Total frames processed: {frame_count}\n")
    f.write(f"Average FPS: {analytics_df['fps'].mean()}\n")
    f.write(f"Total model time: {analytics_df['model_time'].sum()} seconds\n")
    f.write(
        f"Average model time per frame: {analytics_df['model_time'].mean()} seconds\n"
    )
    f.write(f"Total number of faces detected: {analytics_df['num_faces'].sum()}\n")
    f.write(f"Average number of faces per frame: {analytics_df['num_faces'].mean()}\n")
    f.write(f"Average image width: {analytics_df['image_width'].mean()}\n")
    f.write(f"Average image height: {analytics_df['image_height'].mean()}\n")
    f.write(f"Average image channels: {analytics_df['image_channels'].mean()}\n")
