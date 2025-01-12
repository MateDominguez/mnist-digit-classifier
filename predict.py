import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('mnist_model.h5')

# Load the data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess test data
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Make predictions
predictions = model.predict(x_test[:25])

# Visualize predictions
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')

    predicted_number = np.argmax(predictions[i])
    true_number = y_test[i]

    color = 'green' if predicted_number == true_number else 'red'
    plt.title(f'Pred: {predicted_number}\nTrue: {true_number}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()