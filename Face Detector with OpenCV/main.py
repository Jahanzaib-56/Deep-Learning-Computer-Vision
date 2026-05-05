import cv2

face_cascade_path = " "

face_cascade = cv2.CascadeClassifier(face_cascade_path) 

cap = cv2.VideoCapture(1) 

while True:

    ret, frames = cap.read()

    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(grey, 1.5, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frames, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Face Detector", frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
