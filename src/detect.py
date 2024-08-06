import cv2
import numpy as np
import tensorflow as tf
from preprocess import preprocess_image

def detect_damage(image):
    model = tf.keras.models.load_model('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/models/saved_model.h5')
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    return 'Damaged' if prediction > 0.5 else 'Not Damaged'

def capture_and_detect():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            result = detect_damage(frame)
            print(result)
            cv2.imshow('Frame', frame)  # Display the frame with prediction

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
