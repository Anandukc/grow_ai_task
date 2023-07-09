import cv2
import os

# Create a folder to store captured images
name = input("Enter the name of the individual: ")
folder_name = name.replace(" ", "_")
os.makedirs(folder_name, exist_ok=True)

# Initialize OpenCV's face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture facial images
capture_count = 0
camera = cv2.VideoCapture(0)  

while capture_count < 200:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region
        face = frame[y:y+h, x:x+w]

        # Save the captured face image
        image_name = os.path.join(folder_name, f"{capture_count}.jpg")
        cv2.imwrite(image_name, face)

        capture_count += 1

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Capture Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
