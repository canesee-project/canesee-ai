"""
@author: Adel Abu Hashim
"""

import face_recognition as face

def init ():
    """
    Load dataset from datafile .
    """
    import data
    return data.known_faces

def load_test_data ():
    '''
    add test data
    '''
    # load test images
    image_of_adel = face.load_image_file("test_data/adel.jpg")
    image_of_luis = face.load_image_file("test_data/luis.jpg")
    image_of_josh = face.load_image_file("test_data/josh.jpg")
    # extract face encodings for each face in numpy array
    face_encoding_of_adel = face.face_encodings(image_of_adel)[0]
    face_encoding_of_luis = face.face_encodings(image_of_luis)[0]
    face_encoding_of_josh = face.face_encodings(image_of_josh)[0]
    # store each person name and his face encodings-converted from numpy array to list- in dict.
    known_faces = {"adel": face_encoding_of_adel.tolist(),
                   "luis": face_encoding_of_luis.tolist(),
                   "josh": face_encoding_of_josh.tolist()}
    # write dict in data file after updating it
    f = open("data.py", "w")
    f.write("known_faces = {}".format(known_faces))
    f.close()

def new_face (image, name):
    """
       Add new face to data set.
       Args:
           (image- numpy array) image - new person's image
           (str) name - person's name
       """
    # import known_faces from data file.
    known_faces = init()
    image = face.load_image_file(image) #uncomment to test on loaded image file
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
    known_faces = init()
    input_image = face.load_image_file(image) #uncomment to test on loaded image file

    #input_image = image
    # extract face encodings of image in numpy array
    input_image = image
    #input_image_encodings = face.face_encodings(input_image)
    # making two lists; dict values and names.
    known_faces_encodings = list(known_faces.values())
    names = list(known_faces.keys())
    # compare stored face encodings with unknown image encodings.
    for input_image_encoding in input_image_encodings:
        results = face.compare_faces(known_faces_encodings, input_image_encoding)
        # extract index which is true in results list.
        index = int([i for i, x in enumerate(results) if x][0])
    # return the name which is related two right face encodings.
    return names[index]

#load_test_data() #Uncomment to test

##test known image
#print(recognize('test_data/adel.jpg'))







