import cv2

def preprocess_image(image):
    # `image` is a numpy array directly from the webcam
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize to [0, 1]
    return image
