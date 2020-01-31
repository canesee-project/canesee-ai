# Facial Expression Recognition using convolutional neural network.



## Overview
### Requirement
- Python3.6
- opencv
- Keras
- tensorflow-gpu
- tflearn

### Data
[Kaggle_fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data):Include 35587 lableled images, you can download `fer2013.tar.gz` and decompress `fer2013.csv` in the `data` folder.



### Howtouse
- get data
- run ```python data_process.py``` to generate npy files for training
- run ```python train.py``` to train, you will get 3 'Gudi...' files after training
- copy 'Gudi_model_100_epochs_20000_faces.data-00000-of-00001' and rename it 'Gudi_model_100_epochs_20000_faces'
- run ```python predict_pic.py``` to predict faces of images


