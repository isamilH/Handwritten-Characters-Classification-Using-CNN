# Handwritten-Characters-Classification-Using-CNN
This project implements a Convolutional Neural Network (CNN) model to classify Arabic handwritten characters from images. The dataset used contains grayscale images resized to 32x32 pixels, and the model is trained to classify these images into one of 28 possible classes.
## Project Structure
1. Data Loading and Preprocessing:
    The dataset is loaded from image files, and the images are converted to grayscale and resized to 32x32 pixels.
    The labels are extracted from the filenames.
    Data is normalized by dividing pixel values by 255.0 to scale them between 0 and 1.
    One-hot encoding is applied to the labels.

2. Data Augmentation:
    Image augmentation is applied to the training data to improve generalization.
    Techniques like rotation, zooming, width and height shifts, and shear transformations are used.

3. Model Architecture:
    - The model is a sequential CNN with the following layers:
        3 convolutional layers with ReLU activation, followed by batch normalization and max-pooling layers.
        A flattening layer to convert the 2D data to 1D.
        A dense layer with 256 units and ReLU activation, followed by dropout for regularization.
        A final dense layer with a softmax activation for multi-class classification.

4. Model Training:
    The model is compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
    Early stopping and learning rate reduction on plateau callbacks are used to prevent overfitting.
    The model is trained for 50 epochs using the augmented training data, with validation on the test data.

5. Evaluation and Metrics:
    The model's performance is evaluated on the test dataset.
    Accuracy and loss over epochs are plotted for both training and validation sets.
    A classification report is generated showing precision, recall, and F1-score for each class.

6. Model Saving:
    The trained model is saved to an HDF5 file named arabic_handwritten_cnn_model.h5.1
## Dataset
- Training Images: 13,440 images of 32x32 pixels.
- Test Images: 3,360 images of 32x32 pixels.
- Classes: 28 distinct classes representing different handwritten Arabic characters.
## Usage
Prerequisites
    - Python 3.x
    - TensorFlow 2.x
    - NumPy
    - Matplotlib
    - Pillow
    - Scikit-learn

## Evaluation
The model achieves a high accuracy on the test dataset, with detailed metrics provided by the classification report.

## Results
  - Test Loss: Reported after evaluation.
  - Test Accuracy: Approximately 93%.
  - Precision, Recall, and F1-Score:  precision    recall  f1-score   support

           0     1.0000    0.9833    0.9916       120
           1     1.0000    0.9667    0.9831       120
           2     0.8917    0.8917    0.8917       120
           3     0.8138    0.9833    0.8906       120
           4     0.9600    1.0000    0.9796       120
           5     0.9732    0.9083    0.9397       120
           6     0.8939    0.9833    0.9365       120
           7     0.9902    0.8417    0.9099       120
           8     0.8374    0.8583    0.8477       120
           9     0.8692    0.9417    0.9040       120
          10     0.8947    0.8500    0.8718       120
          11     0.9739    0.9333    0.9532       120
          12     0.9677    1.0000    0.9836       120
          13     0.9206    0.9667    0.9431       120
          14     0.9504    0.9583    0.9544       120
          15     0.9712    0.8417    0.9018       120
          16     0.8561    0.9917    0.9189       120
          17     0.9554    0.8917    0.9224       120
          18     0.9664    0.9583    0.9623       120
          19     0.9145    0.8917    0.9030       120
          20     0.9113    0.9417    0.9262       120
          21     0.9200    0.9583    0.9388       120
          22     0.9914    0.9583    0.9746       120
          23     0.9833    0.9833    0.9833       120
          24     0.9439    0.8417    0.8899       120
          25     0.9487    0.9250    0.9367       120
          26     0.9554    0.8917    0.9224       120
          27     0.9440    0.9833    0.9633       120

    accuracy                         0.9330      3360
   macro avg     0.9357    0.9330    0.9330      3360
weighted avg     0.9357    0.9330    0.9330      3360

## Future Improvements
  - Further tuning of the model's hyperparameters.
  - Experimenting with different architectures (e.g., deeper CNN models).
  - Incorporating more advanced data augmentation techniques.
