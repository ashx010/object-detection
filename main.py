import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained object detection model (SSD with MobileNet)
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, you can change it if you have multiple cameras

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to uint8 data type
    input_image = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)

    # Add batch dimension to the input tensor
    input_tensor = tf.expand_dims(input_image, 0)

    # Perform inference
    detections = model(input_tensor)

    # Extract relevant information from the detections
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    # Set a threshold for detection confidence
    threshold = 0.4
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]

            # Convert box coordinates to image coordinates
            h, w, _ = image.shape
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)

            # Draw bounding box on the image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Object Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
