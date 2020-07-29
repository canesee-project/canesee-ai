
from deepface import DeepFace
#model = load_model('model.h5')
#demography = DeepFace.analyze("h.jpg") #passing nothing as 2nd argument will find everything
demography = DeepFace.analyze("h.jpg", ['age', 'gender',  'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print('\n', "-------------------")
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
#print("Race: ", demography["dominant_race"])