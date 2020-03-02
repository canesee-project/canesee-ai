"""
@author: Adel Abu Hashim
"""

import face_recognition as face

known_faces_encodings = []

def init ():
    """Load dataset"""
    global known_faces_encodings

    image_of_adel = face.load_image_file("adel.jpg")
    image_of_luis = face.load_image_file("luis.jpg")
    image_of_josh = face.load_image_file("josh.jpg")

    face_encoding_of_adel = face.face_encodings(image_of_adel)[0]
    face_encoding_of_luis = face.face_encodings(image_of_luis)[0]
    face_encoding_of_josh = face.face_encodings(image_of_josh)[0]

    known_faces_encodings = [face_encoding_of_adel, face_encoding_of_luis, face_encoding_of_josh]

'''def add_face(image, name):
    if (reco(image) == "unknown") :
        face_encodings_of{}.format(name) =face.face_encodings(image)'''


def reco(image):
    """Return a string of the name of person in the photo. """
    global known_faces_encodings

    ##input_image = face.load_image_file(image)
    input_image = image
    input_image_encodings = face.face_encodings(input_image)  # convert photo into array of numbers

    for input_image_encoding in input_image_encodings:
        results = face.compare_faces(known_faces_encodings, input_image_encoding)
        name = "unknown"
        if results[0]:
            name = "adel"
        elif results[1]:
            name = "luis"
        elif results[2]:
            name = "josh"
    return name


init()
my_img = face.load_image_file("luis3.png")
reco(my_img)

