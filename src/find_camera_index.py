import cv2

def find_camera_index():
    max_tested = 10  # You can increase this if you have more cameras connected
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is available.")
            cap.release()
        else:
            print(f"Camera index {i} is not available.")

if __name__ == "__main__":
    find_camera_index()
