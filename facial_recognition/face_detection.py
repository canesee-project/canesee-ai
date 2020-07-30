from PIL import Image
import PIL.ImageDraw
import face_recognition

#i/p image
image = face_recognition.load_image_file('test.png')

#list of lists; each list conatins a locations of face
face_locations_list = face_recognition.face_locations(image)

#A python dictionary which contains the landark name and its coordinates in image
face_landmarks_dict = face_recognition.face_landmarks(image)

num_of_faces = len(face_locations_list)
print(num_of_faces)

#darwing rectangle for each face
pil_image = PIL.Image.fromarray(image)

for face_location in face_locations_list:
    top,right,bottom,left = face_location
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left,top,right,bottom], outline="blue")

for face_landmarks in face_landmarks_dict:
    for name, list_of_points in face_landmarks.items():
        draw.line(list_of_points, fill='green', width=1)

pil_image.show()