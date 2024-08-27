# Image Classifier - Flower Species Recognition

This repository contains the code and resources for an image classifier project developed as part of the Udacity AI Programming with Python Nanodegree. The classifier is designed to recognize different species of flowers using deep learning techniques, particularly Convolutional Neural Networks (CNNs) implemented in PyTorch.

## Project Overview

The objective of this project is to develop a deep learning model capable of classifying images of flowers into 102 different species. This model can be integrated into a broader application, such as a smartphone app that identifies flowers based on user-taken photos.

### Dataset

The project utilizes the [Oxford 102 Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which consists of images of 102 flower categories. The dataset is split into training, validation, and test sets.

## Key Concepts and Techniques

### 1. **Data Loading and Preprocessing**
   - **Image Preprocessing**: 
     - Images are resized, cropped to a uniform size, and normalized. These transformations help improve model performance and generalization.
   - **Data Augmentation**: Techniques such as random rotations, flipping, and scaling are applied to increase the diversity of the training dataset, thereby reducing overfitting.
   - **Data Loaders**: PyTorch's `DataLoader` is used to efficiently load the images in batches, allowing for scalable training and testing.

### 2. **Model Architecture**
   - **Transfer Learning**: 
     - A pre-trained model (VGG16) is used as the base for the classifier. Transfer learning leverages the powerful feature extraction capabilities of deep CNNs trained on large datasets like ImageNet.
   - **Custom Classifier**:
     - The final layer of the pre-trained model is replaced with a custom classifier tailored to the 102 flower categories. The classifier consists of fully connected layers with ReLU activations and dropout for regularization.
   - **Loss Function and Optimizer**:
     - The negative log likelihood loss (`nn.NLLLoss`) is used as the loss function, and the Adam optimizer (`optim.Adam`) is used for training the classifier.

### 3. **Training the Model**
   - **Training Loop**:
     - The model is trained over several epochs, with the loss and accuracy being tracked for both training and validation datasets. The model is optimized using backpropagation and gradient descent.
   - **GPU Acceleration**:
     - If a GPU is available, it is used to accelerate the training process by performing computations on CUDA-enabled devices.

### 4. **Model Evaluation**
   - **Validation**:
     - During training, the model's performance is validated on a separate validation dataset to ensure that it generalizes well to unseen data.
   - **Testing**:
     - After training, the model is evaluated on the test dataset, which the model has never seen before. The test accuracy is expected to be around 70%, depending on the training conditions and hyperparameters.

### 5. **Prediction and Inference**
   - **Image Prediction**:
     - The model can be used to predict the class of new images, returning the most likely flower species along with the probability of each class.
   - **Top-K Classes**:
     - The model outputs the top-K most likely classes for a given input image, providing a ranked list of potential predictions.

## How to Run the Project

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook (optional, for interactive exploration)

### Steps to Run
1. Navigate to the project directory:
   ```bash
   cd image-classifier-flowers
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Image Classifier Project.ipynb"
   ```
4. Alternatively, you can train the model and make predictions using the provided Python scripts:
   - To train the model:
     ```bash
     python train.py --data_dir path_to_data --save_dir path_to_save_model --epochs 5
     ```
   - To make predictions:
     ```bash
     python predict.py --image_path path_to_image --checkpoint path_to_saved_model
     ```

## Results and Interpretation

The model successfully classifies flower images with a reasonable accuracy. The use of transfer learning and data augmentation helps in achieving better performance with limited data. The model's predictions can be used in practical applications such as a flower recognition app.

## Future Work
- **Hyperparameter Tuning**: Experiment with different hyperparameters like learning rate, batch size, and number of epochs to improve model accuracy.
- **Model Architecture**: Explore different pre-trained models (e.g., ResNet, Inception) for potentially better performance.
- **Deployment**: Package the model for deployment in a mobile or web application, allowing real-time flower identification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Udacity for providing the course and guidance.
- The open-source community for maintaining valuable tools like PyTorch.
- The Oxford Visual Geometry Group for the flower dataset.
