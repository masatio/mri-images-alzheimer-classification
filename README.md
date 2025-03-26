# Alzheimer's Disease Classification using Fine-Tuned EfficientNet-b2

This project fine-tunes the EfficientNet-b2 deep learning model for classifying MRI images to detect Alzheimer's Disease. The dataset consists of MRI scans categorized into different stages of Alzheimer's, and we use transfer learning to adapt the EfficientNet-b2 model for this medical imaging task.

## Dataset

Dataset used in this project can be downloaded on this link: https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset
The dataset consists of MRI scans labeled as:
* Non-Demented (Normal)
* Very Mild Demented
* Mild Demented
* Moderate Demented

## Data Preprocessing

Images were resized to 224x224 to match EfficientNet-b2 input. After that, pixel values were normalized and images were converted to tensors. Data was split into train, validation and test sets in ratio 80:20:20.

![image](https://github.com/user-attachments/assets/cbf50b2f-7d19-4c9c-835b-99c69b0b5ce6)

## Model Training

Pretrained EfficientNet-b2 was loaded and slightly modified - the classification layer was replaced by a dropout and a classifying, linear layer with four outputs. 


        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 112, 112]             864
       BatchNorm2d-2         [-1, 32, 112, 112]              64
              SiLU-3         [-1, 32, 112, 112]               0
            Conv2d-4         [-1, 32, 112, 112]             288
       BatchNorm2d-5         [-1, 32, 112, 112]              64
              SiLU-6         [-1, 32, 112, 112]               0
 AdaptiveAvgPool2d-7             [-1, 32, 1, 1]               0
            Conv2d-8              [-1, 8, 1, 1]             264
              SiLU-9              [-1, 8, 1, 1]               0
           Conv2d-10             [-1, 32, 1, 1]             288
          Sigmoid-11             [-1, 32, 1, 1]               0
SqueezeExcitation-12         [-1, 32, 112, 112]               0
           Conv2d-13         [-1, 16, 112, 112]             512
      BatchNorm2d-14         [-1, 16, 112, 112]              32
           MBConv-15         [-1, 16, 112, 112]               0
           Conv2d-16         [-1, 16, 112, 112]             144
      BatchNorm2d-17         [-1, 16, 112, 112]              32
             SiLU-18         [-1, 16, 112, 112]               0
AdaptiveAvgPool2d-19             [-1, 16, 1, 1]               0
           Conv2d-20              [-1, 4, 1, 1]              68
             SiLU-21              [-1, 4, 1, 1]               0
           Conv2d-22             [-1, 16, 1, 1]              80

Only the last three layers are unfrozen and being fine-tuned during the training process. Criterion is crossentropy loss and Adam is used as optimizer with a learning rate of 0.001. The number of training epochs is 15. If there is no improvement in validation loss function over three epochs, early stopping is triggered.

## Results

Accuracy on test set is 98.18%.
![image](https://github.com/user-attachments/assets/7b7100b4-37ed-4d2f-b57c-7f5c5cc16783)

