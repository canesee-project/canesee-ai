import face_recognition as face

image_of_adel = face.load_image_file("adel.jpg")
image_of_azazi = face.load_image_file("azazi.JPG")
face_encoding_of_adel = face.face_encodings(image_of_adel)[0]
face_encoding_of_azazi = face.face_encodings(image_of_azazi)[0]

known_faces_encodings = [face_encoding_of_adel, face_encoding_of_azazi]


def face_recognition_function(image):
    input_image = face.load_image_file(image)
    input_image_encodings = face.face_encodings(input_image)
    for _ in input_image_encodings:
        results = face.compare_faces(known_faces_encodings, input_image_encodings)
        name = "unknown"

        if results[0]:
            name = "adel"
        elif results[1]:
            name = "azazi"
    return name


print(face_recognition_function("ip.jpg"))
