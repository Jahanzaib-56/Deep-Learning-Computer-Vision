import cv2

# Path to Haar Cascade XML file for frontal face detection
face_cascade_path = " "

# Load the pre-trained Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Open video capture — 0 = default webcam, 1 = external camera
cap = cv2.VideoCapture(1)

while True:
    # Read a single frame; ret is False if the frame wasn't captured successfully
    ret, frames = cap.read()

    # Convert frame to grayscale — detectMultiScale works on grayscale images
    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # 1.5 = scaleFactor (how much image size is reduced at each scale)
    # 5   = minNeighbors (higher = fewer but more reliable detections)
    face = face_cascade.detectMultiScale(grey, 1.5, 5)

    # Iterate over each detected face (x, y = top-left corner, w/h = width/height)
    for (x, y, w, h) in face:
        # Draw a blue rectangle around the detected face
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display label slightly above the rectangle
        cv2.putText(frames, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the current frame in a window
    cv2.imshow("Face Detector", frames)

    # Exit the loop when 'q' is pressed
    # waitKey(1) waits 1ms per frame; 0xFF masks to handle 64-bit systems
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
