from train import Build_CNN
import os

model = Build_CNN()
print(model.summary())
weights = './model/emotion_best_weights.h5'
if os.path.exists(weights):
    print("Loading weight")
    model.load_weights(weights)
    print("Saving model")
    model.save("./model/emotion_recognition.h5")
