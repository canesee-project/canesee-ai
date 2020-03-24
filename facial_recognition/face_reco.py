"""
@author: Adel Abu Hashim
"""

import face_recognition as face
import data


known_faces = data.known_faces

def init ():
    '''
    add test data
    '''

    global known_faces

    image_of_adel = face.load_image_file("adel.jpg")
    image_of_luis = face.load_image_file("luis.jpg")
    image_of_josh = face.load_image_file("josh.jpg")

    face_encoding_of_adel = face.face_encodings(image_of_adel)[0]
    face_encoding_of_luis = face.face_encodings(image_of_luis)[0]
    face_encoding_of_josh = face.face_encodings(image_of_josh)[0]

    known_faces = {"adel": face_encoding_of_adel.tolist(),
                   "luis": face_encoding_of_luis.tolist(),
                   "josh": face_encoding_of_josh.tolist()}


    f = open("data.py", "w")
    f.write("known_faces = {}".format(known_faces))
    f.close()



def new_face (image, name):
    """
    updates dict"""
    global known_faces

    image = face.load_image_file(image)
    known_faces[name] = face.face_encodings(image)[0].tolist()
    f = open("data.py", "w")
    f.write("known_faces = {}".format(known_faces))
    f.close()




def reco(image):
    """Return a string of the name of person in the photo. """
    global known_faces
    #input_image = face.load_image_file(image)
    input_image = image
    input_image_encodings = face.face_encodings(input_image)  # convert photo into array of numbers
    known_faces_encodings = list(known_faces.values())
    names = list(known_faces.keys())
    for input_image_encoding in input_image_encodings:
        results = face.compare_faces(known_faces_encodings, input_image_encoding)
        index = int([i for i, x in enumerate(results) if x][0])

    return names[index]



#init() #for first time
#new_face("luis2.png",'new')






