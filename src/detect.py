# import cv2
# import numpy as np
# import tensorflow as tf
# from preprocess import preprocess_image

# def detect_damage(image):
#     model = tf.keras.models.load_model('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/models/saved_model.h5')
#     processed_image = preprocess_image(image)
#     processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
#     prediction = model.predict(processed_image)
#     return prediction < 0.5  # Return True if damaged, False otherwise

# def capture_and_detect():
#     cap = cv2.VideoCapture(0) # index of a camera
#     detecting = True  # Flag to control the detection process

#     while True:
#         if detecting:
#             ret, frame = cap.read()
#             if ret:
#                 is_damaged = detect_damage(frame)
                
#                 if is_damaged:
#                     # Stop detection and show "DAMAGED" message
#                     detecting = False
#                     cv2.putText(frame, "DAMAGED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                 else:
#                     cv2.imshow('Frame', frame)
        
#         else:
#             # Wait for space bar to resume detection
#             cv2.putText(frame, "DAMAGED - Press SPACE to Continue", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#             cv2.imshow('Frame', frame)
            
#             key = cv2.waitKey(1)
#             if key == ord(' '):  # Space bar pressed
#                 detecting = True

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     capture_and_detect()

import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound  # Import playsound to play mp3
from preprocess import preprocess_image

def detect_damage(image):
    model = tf.keras.models.load_model('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/models/saved_model.h5')
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    return prediction < 0.5  # Return True if damaged, False otherwise

def capture_and_detect():
    cap = cv2.VideoCapture(1)  # index of the camera (0 is default)
    
    # Set resolution
    width = 1920  # Change this to your desired width
    height = 1080  # Change this to your desired height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set FPS
    fps = 30  # Increase this to your desired FPS
    cap.set(cv2.CAP_PROP_FPS, fps)

    detecting = True  # Flag to control the detection process

    while True:
        if detecting:
            ret, frame = cap.read()
            if ret:
                is_damaged = detect_damage(frame)

                if is_damaged:
                    # Stop detection, show "DAMAGED" message first
                    detecting = False
                    cv2.putText(frame, "DAMAGED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Frame', frame)  # Show the frame with the "DAMAGED" text
                    cv2.waitKey(500)  # Wait for 500ms before playing the sound
                    # Play your own mp3 alarm sound
                    playsound('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/sounds/alarm.mp3')
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
