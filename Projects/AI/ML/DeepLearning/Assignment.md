Assignment 10: Deep Learning
Completion requirements
Opened: Tuesday, 15 July 2025, 12:00 AM
Due: Monday, 21 July 2025, 11:59 PM
Overview
In this assignment, you will apply your understanding of Artificial Neural Networks and TensorFlow/Keras to build, train, evaluate, and document an image classification model using the MNIST dataset. 

MNIST stands for Modified National Institute of Standards and Technology dataset. It is a benchmark dataset widely used in training and testing machine learning and deep learning models for handwritten digit recognition.


Item	Description
Images	70,000 grayscale images of digits (0–9)
Size	Each image is 28×28 pixels (784 total)
Labels	10 classes (digits 0 to 9)
Split	60,000 training images, 10,000 test images

By completing this assignment, you will demonstrate your ability to:

Preprocess and explore image data
Design and build the ANN architecture
Compile, Train, and Validate the deep learning model
Evaluate the model on the test set and report the final test accuracy 
Visualize model training history
Save and load trained models using the modern Keras format
Instructions 
Complete the following tasks:

Load the MNIST dataset using tensorflow.keras.datasets.
Visualize at least 9 random images with their labels using matplotlib.
Normalize the pixel values to a [0,1] range.
One-hot encode the labels using to_categorical.
Print dataset shapes and confirm correct preprocessing.
Use the Sequential model.
Include at least: Flatten layer as input layer, Two Dense hidden layers (e.g., 128 and 64 neurons) with ReLU activation, Dropout layers (e.g., 0.3) for regularization, Output layer with 10 neurons and softmax activation.

Compile with adam optimizer and categorical_crossentropyloss.

Use accuracy as the evaluation metric.

Train the model for 10 epochs, using a batch_size of 128 and a validation_split of 0.1.

Plot training and validation accuracy/loss per epoch.

Evaluate the model on the test set and report the final test accuracy.

Use model.predict() to get predictions on test data.

Display a confusion matrix using seaborn.heatmap.

Print a classification report showing precision, recall, and F1-score.

Save the trained model in the native Keras format:
Code Snippets
## Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Step 2: Load the Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 3: Preprocess the Data
X_train = X_train / 255.0  # Normalize pixel values to [0,1]
X_test = X_test / 255.0
y_train_cat = to_categorical(y_train, 10)  # One-hot encode labels
y_test_cat = to_categorical(y_test, 10)

# Plot some digits from dataset
selected_indices = [10, 25, 75, 300, 501, 999, 1234, 1500, 1999]  # Choose which image indices to display

plt.figure(figsize=(8, 8))

for i, idx in enumerate(selected_indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(f"Label: {y_train[idx]} (Index: {idx})")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Step 4: Build the ANN Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(X_train, y_train_cat,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1)

# Step 7: Evaluate on Test Set
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 8: Visualize Training History
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 9: Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 10: Classification Report
print(classification_report(y_test, y_pred_classes))

# Step 11: Save and Reload the Model
model.save("mnist_ann_model.h5")

from tensorflow.keras.models import load_model
reloaded_model = load_model("mnist_ann_model.h5")
reloaded_model.evaluate(X_test, y_test_cat)
Submission Guidelines
Work out your assignment on a Python Notebook on Colab. Use markdown and comments to clearly capture your steps as you work. 

In addition, write a report with a cover page that captures your name and program details, followed by the following Sections:

Introduction: What the project is about
Task Completion: Screenshots, explanations, and results
Conclusion: What you learned from the assignment
As you complete your tasks, provide evidence of completion by capturing screenshots with sufficient detail for:

Data loading and exploration
Model training
Evaluation results
Ensure your write-ups and screenshots demonstrate enough detail to confirm your engagement in completing the lab assignment.

Ensure to follow good coding practices by using appropriate names for variables, using comments and white space for code readability.

Share a link to your final Notebook by clicking the Share button on the top-right side of the page. Ensure you allow Public Access

Submit this report as a PDF for marking, and ensure it includes a link to your final work that we can open (test it on an incognito browser)