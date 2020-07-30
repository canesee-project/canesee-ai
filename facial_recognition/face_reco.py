"""
@author: Adel Abu Hashim
adel.muha16@gmail.com
last edit: 30 jul 2020
"""

import face_recognition as face
import data

# fetching stored data
known_faces = data.known_faces


def init():
    """
    Load dataset from datafile .
    """
    global known_faces


def load_test_data():
    """
    add test data
    """
    # load test images
    image_of_adel = face.load_image_file("test_data/adel.jpg")
    image_of_yossef = face.load_image_file("test_data/yossef.jpg")
    image_of_hashim = face.load_image_file("test_data/hashim.jpg")
    # extract face encodings for each face in numpy array
    face_encoding_of_adel = face.face_encodings(image_of_adel)[0]
    face_encoding_of_yossef = face.face_encodings(image_of_yossef)[0]
    face_encoding_of_hashim = face.face_encodings(image_of_hashim)[0]
    # store each person name and his face encodings-converted from numpy array to list- in dict.
    known_faces = {"adel": face_encoding_of_adel.tolist(),
                   "yossef": face_encoding_of_yossef.tolist(),
                   "hashim": face_encoding_of_hashim.tolist()}
    # write dict in data file after updating it
    f = open("data.py", "w")
    f.write("known_faces = {}".format(known_faces))
    f.close()


def new_face(image, name):
    """
       Add new face to data set.
       Args:
           (image- numpy array) image - new person's image
           (str) name - person's name
       """
    # import known_faces from data file.
    global known_faces
    image = face.load_image_file(image)  # uncomment to test on loaded image file
    # add new element to known_faces dict.
    # use "tolist()" to permit importing dict.
    known_faces[name] = face.face_encodings(image)[0].tolist()
    # save appended dict value into data.py file
    f = open("data.py", "w")
    f.write("known_faces = {}".format(known_faces))
    f.close()


def recognize(image):
    """
       Add new face to data set.
       Args:
           (image- numpy array) image - Person's image
        Returns:
            names[index] - Recognized person name.
       """
    # import known_faces from data file.
    global known_faces
    # input_image = face.load_image_file(image)  # uncomment to test on loaded image file (PC)

    # extract face encodings of image in numpy array
    input_image = image  # (PI)
    input_image_encodings = face.face_encodings(input_image)
    # making two lists; dict values and names.
    known_faces_encodings = list(known_faces.values())
    names = list(known_faces.keys())

    # compare stored face encodings with unknown image encodings.
    for input_image_encoding in input_image_encodings:
        results = face.compare_faces(known_faces_encodings, input_image_encoding)

    # extract index which is true in results list.
    try:
        index = int([i for i, x in enumerate(results) if x][0])
        person = names[index]
    except IndexError:
        person = 'new face'
    # return the name which is related to right face encodings.
    return person

# load_test_data() # uncomment to test

# test known image (PC)
# print(recognize('test_data/new.jpg'))
