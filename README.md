# Alzheimer's Disease Classification using Fine-Tuned EfficientNet-b2

This project fine-tunes the EfficientNet-B2 deep learning model to classify MRI images for Alzheimer's Disease detection. The dataset consists of MRI scans categorized into different stages of Alzheimer's, and we leverage transfer learning to adapt the model for this medical imaging task.

## Dataset

The dataset used in this project can be downloaded from the following link: https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset
The dataset consists of MRI scans labeled as:
* Non-Demented (Normal)
* Very Mild Demented
* Mild Demented
* Moderate Demented

## Data Preprocessing

Images were resized to 224x224 to match EfficientNet-b2 input. After that, pixel values were normalized and images were converted to tensors. The dataset was split into training (80%), validation (10%), and test (10%) sets.

![image](https://github.com/user-attachments/assets/cbf50b2f-7d19-4c9c-835b-99c69b0b5ce6)

## Model Training

We used a pretrained EfficientNet-B2 model, modifying its classification head:

* The original classifier was replaced with a dropout layer and a fully connected linear layer with four output classes.

* Only the last three layers were unfrozen and fine-tuned, while the rest of the network remained frozen.


| Layer Type               | Output Shape          | Parameters |
|--------------------------|----------------------|------------|
| **Conv2d**              | `[-1, 32, 112, 112]` | 864        |
| **BatchNorm2d**         | `[-1, 32, 112, 112]` | 64         |
| **SiLU Activation**      | `[-1, 32, 112, 112]` | 0          |
| **Conv2d**              | `[-1, 32, 112, 112]` | 288        |
| **BatchNorm2d**         | `[-1, 32, 112, 112]` | 64         |
| **SiLU Activation**      | `[-1, 32, 112, 112]` | 0          |
| **AdaptiveAvgPool2d**   | `[-1, 32, 1, 1]`     | 0          |
| **Conv2d**              | `[-1, 8, 1, 1]`      | 264        |
| **SiLU Activation**      | `[-1, 8, 1, 1]`      | 0          |
| **Conv2d**              | `[-1, 32, 1, 1]`     | 288        |
| **Sigmoid Activation**   | `[-1, 32, 1, 1]`     | 0          |
| **SqueezeExcitation**    | `[-1, 32, 112, 112]` | 0          |
| **Conv2d**              | `[-1, 16, 112, 112]` | 512        |
| **BatchNorm2d**         | `[-1, 16, 112, 112]` | 32         |
| **MBConv**              | `[-1, 16, 112, 112]` | 0          |
| **Conv2d**              | `[-1, 16, 112, 112]` | 144        |
| **BatchNorm2d**         | `[-1, 16, 112, 112]` | 32         |
| **SiLU Activation**      | `[-1, 16, 112, 112]` | 0          |
| **AdaptiveAvgPool2d**   | `[-1, 16, 1, 1]`     | 0          |
| **Conv2d**              | `[-1, 4, 1, 1]`      | 68         |
| **SiLU Activation**      | `[-1, 4, 1, 1]`      | 0          |
| **Conv2d**              | `[-1, 16, 1, 1]`     | 80         |

## Training Details 

* Loss Function: Cross-Entropy Loss
* Optimizer: Adam with a learning rate of 0.001
* Epochs: 15
* Early Stopping: If the validation loss does not improve for 3 consecutive epochs, training is stopped.

## Results

Accuracy on test set is 98.18%.
![image](https://github.com/user-attachments/assets/7b7100b4-37ed-4d2f-b57c-7f5c5cc16783)

