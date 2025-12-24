import cv2
import os

# List of images
image_list = [
    "images/astronaut1.jpg",
    "images/space1.jpg",
    "images/tools1.jpg"
]

for path in image_list:
    img = cv2.imread(path)

    if img is None:
        print("Image not found:", path)
        continue

    # Draw rectangle (example unsafe zone)
    cv2.rectangle(img, (100, 100), (400, 300), (0, 0, 255), 3)

    # STATUS text
    cv2.putText(
        img,
        "STATUS: CHECKING SAFETY",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # DANGER text
    cv2.putText(
        img,
        "DANGER: UNSAFE OBJECT",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Show image
    cv2.imshow("Space Safety Check", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()