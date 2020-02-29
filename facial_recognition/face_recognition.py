import face_recognition as face

image_of_juno = face.load_image_file("juno.JPG")
image_of_luis = face.load_image_file("luis.jpg")
image_of_josh = face.load_image_file("josh.jpg")
face_encoding_of_juno = face.face_encodings(image_of_juno)[0]
face_encoding_of_luis = face.face_encodings(image_of_luis)[0]
face_encoding_of_josh = face.face_encodings(image_of_josh)[0]

known_faces_encodings = [face_encoding_of_juno, face_encoding_of_luis,face_encoding_of_josh]


def face_recognition_function(image):
    """Return a string of the name of person in the photo. """

    input_image = face.load_image_file(image)
    input_image_encodings = face.face_encodings(input_image)  # convert photo into array of numbers
    for input_image_encoding in input_image_encodings:

        results = face.compare_faces(known_faces_encodings, input_image_encoding)

        name = "unknown"

        if results[0]:
            name = "juno"
        elif results[1]:
            name = "luis"
        elif results[2]:
            name = "josh"
    return name


print(face_recognition_function("juno2.jpg"))
