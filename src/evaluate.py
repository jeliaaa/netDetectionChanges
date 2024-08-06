import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model():
    model = tf.keras.models.load_model('C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/models/saved_model.h5')

    datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = datagen.flow_from_directory(
        'C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/data/validation',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    evaluate_model()
