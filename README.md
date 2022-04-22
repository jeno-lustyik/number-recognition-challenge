# number-recognition-challenge
 Jeno / Nagarjuna // Strive School AI_JAN_22


#### Introduction
The Idea behind this project is to predict the hand written digits of a user by CNN model.  
We trained the model with available MNIST dataset in the torchvision library.
We defined a Image preprocess function that takes hand written image from the user and preprocess it.
The preprocessed image will act as input to our model for prediction.

#### What to Install
pip install -requirements.txt

#### Technologies/libraries used
Python == 3.8.13
PyTorch == 1.11
Opencv-python == 4.5.5  
numpy == 1.14.6  
matplotlib == 3.2.2

#### How to use it
1. First Install the required packages.
2. There was a folder named images to place your hand written digit images.
3. Run `main.py` file.
4. It gonna ask you image input in the terminal.
5. Enter the image name to be predicted with extension.
6. In the end it gonna give you, your input image and the predicted output.

#### Info on files
1. processing.py --- It has a pipeline that can preprocess the user input image.
2. data_handler.py --- Contains the code that can downlad training data from the Trochvision.
3. model.py --- Contains CNN model for predicting Hand written digits.
4. training.py --- Contains the training loop to train the model and predict output.
5. helper_fun.py --- It has some helper functions to plot images and output.
6. main.py --- It is the main file where user can input and predict.

#### Demo.
User Input Image Example.
<img src= ".\images\numbers.jpg" alt="Image with a Green Screen Background. "/>   
Images look like after preprocessing
<img src= ".\images\3.jpg" alt="Image with a Green Screen Background. "/>  
Train vs Test Losses during training. 
<img src= ".\images\trainvstest_losses.JPG" alt="Image with a Green Screen Background. "/>    
  
Accuracy of the model.  
<img src= ".\images\Accuraccy.JPG" alt="Image with a Background."/>    


Final output   
<img src= ".\images\final output.JPG" alt="Final Image"/> 
  