import cv2
import numpy as np
import time  # add this

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frames = []
gap = 5
count = 0
last_saved = 0      # track when we last saved
cooldown = 3        # seconds between saves

while True:
    ret, frame = cap.read()

    time.sleep(0.03)

    if not ret:
        print("Failed to Capture.....")
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames.append(grey)  # Bug 1 fix

    if len(frames) > gap + 1:  # Bug 2 fix
        frames.pop(0)

    cv2.putText(frame, f'Frame Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    count += 1  # only once now (Bug 5 fix)

    if len(frames) > gap:  # Bug 2 fix
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        motion = any(cv2.contourArea(c) >= 1000 for c in contours)  # Bug 4 fix

        if motion:
            cv2.putText(frame, "Motion Detected!!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            now = time.time()
            if now - last_saved > cooldown:  # only save if cooldown has passed
                cv2.imwrite(f"Motion_frame_{count}.jpg", frame)
                print(f"Saved: Motion_frame_{count}")
                last_saved = now

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  # Bug 6 fix
cv2.destroyAllWindows()  # Bug 6 fix
