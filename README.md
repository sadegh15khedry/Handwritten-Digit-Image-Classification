# MNIST Digit Classification

This project implements a neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model architecture includes several dense layers, batch normalization, and dropout for regularization.

- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Acknowledgments](#acknowledgments)
- [Further Improvements](#further-improvements)
- [License](#license)

## Directory Structure
```
├── src
│ ├── utils.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ └── model_evaluation.ipynb
├── models
│ ├── model_custom.joblib
│ └── model_sklearn.joblib
├── environment.yml
└── README.md
```
## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `model_training.py` : Functions for training the model.
- `model_evaluation.py` : Functions for evaluating the model.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for model training.
- `model_evaluation.ipynb`: Notebook for model evaluation.

   
## Dataset

The MNIST dataset is used in this project. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).


## Model Performance

### Train and Validation data Results

- train loss: 0.0616
- train accuracy: 0.9844
- val_loss: 0.0666
- val_accuracy: 0.9859

![training_validation_loss_and_accuracy](https://github.com/user-attachments/assets/d7384c5f-4fbe-48eb-a019-69c78740290b)
*Train accuracy and loss*


# Test Data Results

- Test accuracy: 0.9872307777404785
- Test loss: 0.06959925591945648
- precision: 0.9872685356098663
- recall: 0.9872307985515533
- class based reesults
```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99      1035
           1       0.99      1.00      0.99      1181
           2       0.99      0.98      0.99      1048
           3       0.99      0.99      0.99      1071
           4       0.98      0.99      0.98      1023
           5       0.98      0.99      0.99       946
           6       0.98      1.00      0.99      1031
           7       0.99      0.99      0.99      1093
           8       0.99      0.98      0.98      1023
           9       0.99      0.98      0.98      1043

    accuracy                           0.99     10494
   macro avg       0.99      0.99      0.99     10494
weighted avg       0.99      0.99      0.99     10494
```

![cm](https://github.com/user-attachments/assets/68a2bb7c-6bc6-4d1e-b8f6-09a91578c1eb)
*Confiusion matrix*

## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/Handwritten-Digit-Image-Classification.git
    cd Handwritten-Digit-Image-Classification
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate minst
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```

## Acknowledgments

- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, Matplotlib and tensorFlow.
- Huge thaks to contributors of the MINST Dataset.

## Further Improvements

- Experimenting models like VGG, ResNet, Inception and etc

# License
This project is licensed under the MIT License - see the LICENSE file for details.














