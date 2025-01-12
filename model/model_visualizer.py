import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ModelVisualizer:
    '''Class for visualizing and analizing neural network model performance and internals.'''
    subplot_height = 2.5
    subplot_width = 2

    def __init__(self, model, x_train, y_train, x_test, y_test):
        '''Initialize the visualizer with model and data.'''
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def visualize_layer_outputs(self, image_index=0):
        '''Visualize the ouput of each convolutional layer for a single image.'''
        # Get all convolutional layers
        conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name]
        
        # Create list of models, one for each conv layer
        layer_outputs = []
        for layer in conv_layers:
            intermediate_model = tf.keras.Model(inputs=self.model.input,
                                                outputs=layer.output,
                                                name=f'intermediate_{layer.name}')
            layer_outputs.append(intermediate_model.predict(self.x_test[image_index:image_index + 1]))

        # Plot the orignal image and layer activations
        width = len(layer_outputs) * self.subplot_width
        height = self.subplot_height
        plt.figure(figsize=(width, height))

        # Plot original image
        plt.subplot(1, len(layer_outputs) + 1, 1)
        plt.imshow(self.x_test[image_index].reshape(28, 28), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Plot each convolutional layer's activiation
        for i, activation in enumerate(layer_outputs):
            plt.subplot(1, len(layer_outputs) + 1, i + 2)
            # Display the first channel of each conv layer
            plt.imshow(activation[0, :, :, 0], cmap='viridis')
            plt.title(f'Conv layer {i + 1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        '''Plot confusion matrix of model predictions.'''
        # Get predictions
        y_pred = np.argmax(self.model.predict(self.x_test, verbose=0), axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Plot with seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        # Print classification report
        print('\nClassification Report:')
        print(classification_report(self.y_test, y_pred))

    def plot_misclassified_examples(self, num_examples=10):
        '''Display misclassified examples.'''
        predictions = self.model.predict(self.x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Find misclassified examples
        misclassified = np.where(predicted_classes != self.y_test)[0]

        # Plot some misclassified examples
        num_examples = min(num_examples, len(misclassified))

        width = num_examples * self.subplot_height
        height = self.subplot_height
        plt.figure(figsize=(width, height))

        for i in range(num_examples):
            idx = misclassified[i]
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(self.x_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'Pred: {predicted_classes[idx]}\nTrue: {self.y_test[idx]}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_feature_maps_animation(self, image_index=0):
        '''
        Plot all feature maps from each convolutional layer.
        Shows how different filters in each conv layer respond to the input image.

        Args:
            image_index (int): Index of the image to analize
        '''
        try:   
            # Get outputs of all convolutional layers
            layer_outputs = [layer.output for layer in self.model.layers if 'conv' in layer.name]

            # Create a new model that will output the activations of conv layers
            activation_model = tf.keras.Model(inputs=self.model.input, outputs=layer_outputs)

            # Get the activation (feature maps) of the image 
            activations = activation_model.predict(self.x_train[image_index:image_index + 1], verbose=0)

            # Plot feature maps for each convolutional layer
            for i, activation in enumerate(activations):
                # Calculate grid size based on the number of features
                num_features = activation.shape[-1] # Number of filters in this layer
                size = int(np.ceil(np.sqrt(num_features))) # Square grid size

                plt.figure(figsize=(15, 8))

                # Plot each feature map
                for j in range(num_features):
                    plt.subplot(size, size, j + 1)

                    # Get the feature map for current filter
                    feature_map = activation[0, :, :, j]

                    # Plot with a consisten color scale for better comparison
                    vmin, vmax = feature_map.min(), feature_map.max()
                    plt.imshow(feature_map, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.axis('off')

                plt.suptitle(f'Feature Maps of Convolutional Layer {i+1}\n'
                             f'({num_features} filters)', fontsize=12)
                plt.tight_layout()
                plt.show()

                # Print information about the layer's feature maps
                print(f'\nLayer {i+1} details:')
                print(f'- Number of feature maps: {num_features}')
                print(f'- Feature map shape: {activation.shape[1]}x{activation.shape[2]}')


        except Exception as e:
            print(f'Error generating feature maps: {str(e)}')
            raise