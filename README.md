# Cat-and-Dog-Image-Classifier

# Cat and Dog Image Classifier

This repository contains a Jupyter Notebook for building and training a convolutional neural network (CNN) to classify images of cats and dogs. The project leverages TensorFlow and Keras libraries for deep learning.

## About the Project

The goal of this project is to create a model that can accurately distinguish between images of cats and dogs. The project follows these key steps:

1. **Dataset Preparation**: 
   - The dataset is sourced from Kaggle's Dogs vs. Cats dataset.
   - The images are organized into training and testing directories.

2. **Data Preprocessing**: 
   - Images are resized to a standard size.
   - Data augmentation techniques are applied to improve the robustness of the model.

3. **Model Architecture**: 
   - A convolutional neural network (CNN) is designed using Keras.
   - The architecture includes several convolutional layers followed by max-pooling layers, and fully connected layers at the end.

4. **Model Training**: 
   - The model is trained using the training dataset.
   - Hyperparameters such as learning rate, batch size, and number of epochs are optimized.

5. **Evaluation and Prediction**: 
   - The trained model is evaluated on the test dataset.
   - Performance metrics such as accuracy, precision, recall, and F1-score are calculated.
   - The model's predictions are visualized.

## Getting Started

### Prerequisites

To run the notebook, you need the following dependencies:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install the dependencies using pip:

```sh
pip install tensorflow keras numpy pandas matplotlib
