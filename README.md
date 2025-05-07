# ASL Alphabet Recognition – Model README

## Overview

### Background

Sign language serves as a crucial communication tool for the deaf and hard-of-hearing community. However, most people outside these communities are not familiar with sign language, creating a gap in communication. This highlights the need for automated systems capable of translating sign language into written or spoken formats. In the United States, American Sign Language (ASL) is the most widely adopted variant. It often uses individual hand signs to represent letters—a process known as fingerspelling—e...

### Objective

This project aims to build a deep learning-based image classification system capable of recognizing American Sign Language (ASL) alphabet signs used in fingerspelling. Given a single image of a hand gesture, the model should accurately identify the corresponding ASL letter. The project further explores multiple deep learning architectures to identify the most effective model and assess its generalization performance on unseen ASL samples.

## Dataset

This project leverages two public American Sign Language (ASL) datasets from Kaggle:

- **Primary training dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
  - Contains 87,209 RGB images in JPEG format.
  - Image resolution: 200×200 pixels.
  - Located after download at: `./asl_alphabet_train/asl_alphabet_train/`
  - Includes 29 folders, representing:
    - The 26 English letters (A–Z)
    - "space", "del", and "nothing"
  - Each folder contains ~3000 labeled images.

- **Supplementary test dataset**: [ASL Alphabet Test Dataset](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)  
  - Located after download at: `./archive/`
  - Each class folder (A–Z, space, del, nothing) contains **30** test images (total: 870).
  - Initially, the archive folder includes a duplicate folder named `asl-alphabet-test` that replicates the same 29 classes.
  - To prevent incorrect evaluation due to class duplication, the `asl-alphabet-test` folder is deleted after download.

This dual-dataset approach enables robust evaluation of the model’s performance on both controlled and real-world inputs.

## Running the Model

1. **Initialize the dataset**:
   - Run `init.ipynb` to download both datasets (ASL Alphabet & ASL Alphabet Test) from Kaggle.
   - Kaggle authentication is required (username + API token).
   - The downloaded datasets will be saved to:
     - Training set: `./asl_alphabet_train/asl_alphabet_train/`
     - Supplementary test set: `./archive/`

2. **Train and evaluate**:
   - Run any `model_{name}.ipynb` script (e.g., `model_googlenet.ipynb`) from the root directory.
   - Each script:
       - Is fully self-contained
       - Defines model structure, data pipeline, training loop, and evaluation steps
         - Has editable hyperparameters at the top of the notebook
   - All scripts list:
       - model_MLP.ipynb
       - model_CNN_Baseline.ipynb
       - model_mobilenetv2.ipynb
       - model_ResNet18.ipynb
       - model_googlenet.ipynb



3. **Important**:
   - Ensure dataset paths are correctly set in each script to match your local file system.
   - Redundant `asl-alphabet-test` folder is deleted automatically to avoid label confusion during evaluation.

## Structure

All scripts and datasets are located in the root project directory. The structure is as follows:

```
FinalProject/
├── init.ipynb                     # Downloads and prepares datasets
├── model_{name}.ipynb             # Example model script (others follow same pattern)
├── archive/                      # Contains real-world test images (after removing asl-alphabet-test)
│   ├── A/
│   ├── B/
│   └── ... (29 folders total)
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ... (29 folders total, each with ~3000 images)
```

There are **no external utility scripts** (`data.py`, `eval.py`, etc.). All logic is implemented directly inside the model notebooks.

## License & Attribution

Datasets sourced from:

- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [ASL Alphabet Test Dataset](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)

This project is for academic and non-commercial use only.



## Model Script Structure

Each model notebook (`model_googlenet.ipynb`, `model_mobilenetv2.ipynb`, `model_resnet18.ipynb`, `model_baseline_cnn.ipynb`, `model_mlp.ipynb`) follows a unified structure composed of the following key stages:

1. **Image Preprocessing**  
   - Define and apply image transformations using `torchvision.transforms`  
   - Use augmentation techniques such as `RandomResizedCrop`, `ColorJitter`, `RandomRotation`, and normalization for training data  
   - Apply consistent resizing and normalization for validation and test data  

2. **Load and Partition the Dataset**  
   - Load images using `torchvision.datasets.ImageFolder`  
   - Split the primary dataset into training, validation, and test subsets using `random_split`  
   - Create `DataLoader` objects for efficient batch loading  

3. **Defining the Model**  
   - Instantiate the model architecture (custom CNN, MLP, or pretrained model)  
   - Modify the final layer to support 29 output classes

4. **Training Function**  
   - Define the training loop with forward pass, loss computation, backpropagation, and optimization  
   - Incorporate validation within each epoch and record accuracy and loss  
   - Save the best-performing model based on validation accuracy  

5. **Drawing Functions**  
   - Visualize loss and accuracy curves over training epochs using `matplotlib`  
   - Compare training and validation performance trends  

6. **Model Evaluation**  
   - Reload the best model checkpoint  
   - Run inference on the held-out test set  
   - Compute and display standard evaluation metrics (Accuracy, Precision, Recall, F1 Score)  

7. **Confusion Matrix Visualization**  
   - Generate a normalized confusion matrix using `seaborn.heatmap`  
   - Provide class-level error distribution for model interpretability  

8. **Model Evaluation on Supplementary Dataset** *(Only for GoogLeNet)*  
   - Evaluate GoogLeNet's generalization on an external dataset located in `./archive/`  
   - Filter out the `asl-alphabet-test` folder due to redundancy  
   - Compute evaluation metrics and visualize confusion matrix  
