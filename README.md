# MNIST Digit Classification

This project implements a neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model architecture includes several dense layers, batch normalization, and dropout for regularization.

## Project Structure

- `docs/`: Documentation and references.
- `results/`: Results and model outputs.
- `src/`: Source code for the project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experiments.
## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sadegh15khedry/Handwritten-Digit-Image-Classification
   cd Handwritten-Digit-Image-Classification
   ```

2. Install the required libraries using the environment.yml file using conda:
   ```bash
   conda env create -f environment.yml
   ```
   
## Dataset

The MNIST dataset is used in this project. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).


## Training

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. It is trained for 30 epochs with a batch size of 64. Data augmentation is applied to improve generalization.

![training_validation_loss_and_accuracy](https://github.com/sadegh15khedry/Handwritten-Digit-Image-Classification/assets/90490848/89e8ac94-d301-4853-b1cf-0e60068f1c1a)

Evaluation
The model's performance is evaluated using precision, recall, and F1-score metrics for each class. The current performance indicates the need for further tuning and experimentation to improve accuracy.
 
 precision    recall  f1-score   support

           0       0.10      0.10      0.10      1035
           1       0.11      0.11      0.11      1181
           2       0.08      0.08      0.08      1048
           3       0.11      0.11      0.11      1071
           4       0.09      0.09      0.09      1023
           5       0.09      0.09      0.09       946
           6       0.11      0.11      0.11      1031
           7       0.10      0.10      0.10      1093
           8       0.10      0.10      0.10      1023
           9       0.10      0.10      0.10      1043

    accuracy                           0.10     10494
   macro avg       0.10      0.10      0.10     10494
weighted avg       0.10      0.10      0.10     10494

License
This project is licensed under the MIT License - see the LICENSE file for details.

