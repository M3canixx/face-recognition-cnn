import dlib
import cv2
from imutils import face_utils
#Renvoie l'image du visage d'une personne sur l'image entrée en paramètre.
def main(image):
    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)
    if len(faces) == 1:
        for face in faces:
            face_bounding_box = face_utils.rect_to_bb(face)
            if all(i >= 0 for i in face_bounding_box):
                [x, y, w, h] = face_bounding_box
                frame = image[y:y + h, x:x + w]
                frame = cv2.resize(frame, (224, 224))
                return frame
    else:
        return 0
    return 0