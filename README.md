# Fruit-ANN-CNN
![Fruit-ANN-CNN Deep Learning Project](https://github.com/Ali-jalil88/COVID-19-Xray-Dataset--CNN-ResNet50-MobileNetV2/blob/main/DALL%C2%B7E%202024-10-07%2014.26.36%20-%20An%20illustration%20comparing%20three%20deep%20learning%20models_%20CNN%2C%20ResNet50%2C%20and%20MobileNetV2.%20Each%20model%20should%20be%20represented%20by%20a%20simplified%20diagram%2C%20showin.webp)
### For a fruit classification task using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) in Keras with TensorFlow, the workflow involves several key components such as data loading, preprocessing (using ImageDataGenerator), model building (ANN and CNN), choosing optimizers (Adam), setting loss functions, and evaluating the model using classification metrics.
#### 1. Dataset Preparation
You can use a publicly available Fruit Classification dataset, such as the Fruit 360 dataset from Kaggle. This dataset includes images of various fruits, which are typically split into training and test sets.
#### 2. Data Loading and Preprocessing
We'll use ImageDataGenerator for preprocessing the images, such as rescaling, augmenting, and splitting into batches.
#### 3. Building an ANN Model
An ANN model may not be ideal for image classification, but here is how you can build one for comparison
#### 4. Building a CNN Model
For image classification, CNN is more effective as it can capture spatial hierarchies in images. Here’s a simple CNN model
#### 5. Training the Model
We will now train both the ANN and CNN models using the fit method.
#### 6. Evaluating the Model
You can evaluate the models using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. To plot and view these, Keras' model.evaluate() and classification_report can be used.
#### 7. Metrics and Optimizers
The Adam optimizer is widely used for classification tasks in CNNs due to its adaptive learning rate. Metrics like accuracy can be tracked during training and visualized afterward using training history.
#### 8. Choosing Loss Functions
Since this is a multi-class classification task, the appropriate loss function is categorical_crossentropy. For binary classification, you would use binary_crossentropy.
#### Summary of Steps:
- Data Preprocessing: Use ImageDataGenerator for augmentation and scaling.
- Model Building: Create ANN and CNN models.
- Optimizer: Use Adam optimizer for adaptive learning.
- Metrics: Track accuracy and use classification metrics like precision, recall, F1-score, and confusion matrix for model evaluation.
## Links:
- **[Project Notebook](https://www.kaggle.com/code/alialarkawazi/fruit-ann-cnn)**
- **[Dataset](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition)**
