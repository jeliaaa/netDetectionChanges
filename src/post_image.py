import cv2
import numpy as np
import tensorflow as tf
import requests
from preprocess import preprocess_image
import io

def detect_damage(image):
    model = tf.keras.models.load_model('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/models/saved_model.h5')
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    return prediction > 0.5  # Return True if damaged, False otherwise

def send_damage_data_with_image(image):
    # Encode image as JPEG
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = io.BytesIO(img_encoded)

    # Prepare payload
    files = {'image': ('frame.jpg', img_bytes, 'image/jpeg')}
    data = {'status': 'damaged'}

    try:
        response = requests.post('https://link.com/net', files=files, data=data)
        if response.status_code == 200:
            print("Data and image sent successfully!")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

def capture_and_detect():
    cap = cv2.VideoCapture(0) # index of a camera
    detecting = True  # Flag to control the detection process

    while True:
        if detecting:
            ret, frame = cap.read()
            if ret:
                is_damaged = detect_damage(frame)
                
                if is_damaged:
                    # Stop detection and show "DAMAGED" message
                    detecting = False
                    send_damage_data_with_image(frame)  # Send data and image to the API
                    cv2.putText(frame, "DAMAGED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.imshow('Frame', frame)
        
        else:
            # Wait for space bar to resume detection
            cv2.putText(frame, "DAMAGED - Press SPACE to Continue", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space bar pressed
                detecting = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
