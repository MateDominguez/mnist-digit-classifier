from datetime import datetime
from model import MNISTClassifier, ModelVisualizer
import tensorflow as tf

def main():
    # Load and preprocess data
    print('Loading and preprocessing data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Create and train the model
    print('Training model...')
    classifier = MNISTClassifier()
    classifier.build_model(dropout_rate=0.3)
    classifier.compile_model(learning_rate=0.001)
    classifier.train(x_train, y_train, epochs=15)

    # Create visualizer
    print('Creating visualizations...')
    visualizer = ModelVisualizer(classifier.model, x_train, y_train, x_test, y_test)

    # Generate visualizartions
    visualizer.visualize_layer_outputs(image_index=42)
    visualizer.plot_confusion_matrix()
    visualizer.plot_misclassified_examples()
    visualizer.plot_feature_maps_animation(image_index=42)

    # Evaluate on test set
    test_loss, test_accuracy = classifier.model.evaluate(x_test, y_test)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')

    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = f'mnist_model_{timestamp}.h5' 
    classifier.model.save(save_path)
    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    main()