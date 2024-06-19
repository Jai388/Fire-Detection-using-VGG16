from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model(r'C:\Users\Jaiganesh\AppData\Local\Programs\Python\Python311\Optimized_model.h5')
def preprocess_image(image):
    # Resize the image to the size expected by the model (e.g., 64x64)
    img = cv2.resize(image, (224,224))
    # Scale pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    # Expand dimensions to match the input shape (1, 64, 64, 3)
    img = np.expand_dims(img, axis=0)
    return img
def detect_fire(frame, model):
    # Preprocess the frame
    img = preprocess_image(frame)
    # Predict using the trained model
    prediction = model.predict(img)
    # Return True if fire is detected, else False
    return prediction[0][0] > 0.5

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect fire in the frame
    if detect_fire(frame, model):
        cv2.putText(frame, 'Fire Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Fire Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
