import imgaug.augmenters as iaa
import cv2
import os

# Define a list of augmenters including color changes
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),                     # Horizontal flip 50% of the time
    iaa.Affine(rotate=(-25, 25)),        # Rotate images randomly
    iaa.Multiply((0.8, 1.2)),            # Random brightness adjustment
    iaa.AddToHueAndSaturation((-10, 10)), # Random hue and saturation changes
    iaa.GaussianBlur(sigma=(0, 3.0)),    # Apply Gaussian blur
    iaa.Grayscale(alpha=(0.0, 1.0)),     # Convert images to grayscale with probability
    iaa.GammaContrast((0.5, 2.0))        # Change contrast by gamma adjustment
])

def augment_and_save(input_path, output_path, num_augmentations=5):
    for img_name in os.listdir(input_path):
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
            image = cv2.imread(os.path.join(input_path, img_name))
            
            for i in range(num_augmentations):
                augmented_image = augmenters(image=image)
                cv2.imwrite(os.path.join(output_path, f"aug_{i}_{img_name}"), augmented_image)

# Call the function to augment images
augment_and_save('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/data/validation/damaged', 'C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/data/validation/damaged', num_augmentations=5)
