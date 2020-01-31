import PIL.Image
import PIL.ImageDraw
import face_recognition


image = face_recognition.load_image_file("image")
face_locations = face_recognition.face_locations(image)

num_of_faces = len(face_locations)
print(num_of_faces)

pil_image = PIL.Image.formarray(image)

for face_location in face_locations:
    top,right,bottom,left = face_location
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left,top,right,bottom], outline="green")

pil_image.show()

