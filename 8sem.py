import cv2

# Replace '1' with the correct index for your USB camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open USB camera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("USB Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Failed to capture image")
        break

cap.release()
cv2.destroyAllWindows()