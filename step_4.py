import cv2
from keras.models import load_model
import joblib
import numpy as np

# Load the trained models
cnn_model_path = r'D:\computer_vission\task_growai\cnn_face_recognition_model'
svm_model_path = r'D:\computer_vission\task_growai\svm_face_recognition_model.pkl'
cnn_classifier = load_model(cnn_model_path)
svm_classifier = joblib.load(svm_model_path)

# Preprocess the input image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input shape of the CNN model
    resized = cv2.resize(gray, (64, 64))
    # Normalize the image pixel values
    normalized = resized / 255.0
    # Expand dimensions to create a batch of size 1
    preprocessed = np.expand_dims(normalized, axis=-1)
    return preprocessed

# Load the pre-trained face detection model
face_cascade_path = r'D:\computer_vission\task_growai\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Function to perform face recognition
def recognize_faces(image):
    # Detect faces in the image using the face detection model
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the image
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess the face region
        preprocessed_face = preprocess_image(face_roi)
        
        # Pass the preprocessed face through the CNN classifier model for prediction
        cnn_prediction = cnn_classifier.predict(preprocessed_face)
        
        # Get the predicted class label from the CNN classifier
        cnn_predicted_class = np.argmax(cnn_prediction)
        
        # Preprocess the face region for the SVM classifier
        flattened_face = preprocessed_face.flatten().reshape(1, -1)
        
        # Pass the flattened face through the SVM classifier model for prediction
        svm_prediction = svm_classifier.predict(flattened_face)
        
        # Get the predicted class label from the SVM classifier
        svm_predicted_class = svm_prediction[0]
        
        # TODO: Define a mapping of class labels to person names
        
        # Draw a bounding box around the detected face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the predicted person name or class label
        # TODO: Use the mapping defined above to display the person name
        
    return image

# input image path
input_image_path = r'D:\computer_vission\task_growai\anandu_kc\split_data\test\4.jpg'

# Load the input image
image = cv2.imread(input_image_path)

# Perform face recognition on the image
result = recognize_faces(image)

# Display the result
#cv2.imshow('Face Recognition Result', result)
if result is not None:
    cv2.imshow('Face Recognition Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face found in the image.")

cv2.waitKey(0)
cv2.destroyAllWindows()
