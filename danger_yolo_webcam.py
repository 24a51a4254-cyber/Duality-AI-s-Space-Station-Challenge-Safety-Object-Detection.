import cv2
import numpy as np

# ------------------ LOAD YOLO FILES ------------------
net = cv2.dnn.readNet(
    "yolo/yolov3.weights",
    "yolo/yolov3.cfg"
)

with open("yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------------ START WEBCAM ------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

danger_objects = ["knife", "scissors", "gun"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ------------------ YOLO PROCESS ------------------
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    detections = net.forward(output_layers)

    danger_detected = False

    for output in detections:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]

                if label in danger_objects:
                    danger_detected = True

                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                color = (0, 0, 255) if label in danger_objects else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # ------------------ STATUS DISPLAY ------------------
    if danger_detected:
        cv2.putText(
            frame,
            "STATUS: DANGER",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
    else:
        cv2.putText(
            frame,
            "STATUS: SAFE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("AI Space Station Safety Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()