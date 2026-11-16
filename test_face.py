
import cv2
from collections import deque, Counter
from face_detection import detect_expression
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# Store the last 5 predictions
prediction_history = deque(maxlen=5)

current_expression = "Detecting..."
current_confidence = 0

last_detection_time = 0
detection_interval = 0.3  # 300 milliseconds

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    current_time = time.time()

    # Run detection every 300 milliseconds
    if current_time - last_detection_time >= detection_interval:

        result = detect_expression(frame)

        expression = result["expression"]
        confidence = result["confidence"]

        if expression != "No face detected":
            prediction_history.append((expression, confidence))
        else:
            prediction_history.clear()
            current_expression = "No face detected"
            current_confidence = 0

        if prediction_history:

            # Find the most common expression
            expressions = [item[0] for item in prediction_history]

            current_expression = Counter(
                expressions
            ).most_common(1)[0][0]

            # Average confidence for that expression
            matching_confidences = [
                item[1] for item in prediction_history
                if item[0] == current_expression
            ]

            current_confidence = sum(
                matching_confidences
            ) / len(matching_confidences)

        last_detection_time = current_time

    # Display stable prediction
    cv2.putText(
        frame,
        f"Expression: {current_expression}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {current_confidence:.1f}%",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("SignSense - Facial Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
