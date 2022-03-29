import os
import matplotlib.pyplot as plt
import getCroppedFace
import cv2
def main(name, brut_path, resize_path):
    total_pictures = 0
    detected_faces = 0
    pathToFolder = brut_path + name
    images_files = os.listdir(pathToFolder) #Liste de toutes les images.jpg de cette personne.
    number_of_images = len(images_files)
    face_images = []
    face_images_files_names = []
    for image_number in range(number_of_images):
        total_pictures += 1
        image_file_name = str(images_files[image_number])
        image_real_number = image_file_name.split(' ')[-1].split('.')[0] #On recupere le numero de l'image à partir du nom du fichier.
        image_path = pathToFolder + '/' + image_file_name
        image = plt.imread(image_path)
        cropped_face = getCroppedFace(image) #On récupère le visage sur la photo.
        
        if type(cropped_face) != int: #Si un seul visage a été trouvé sur la photo.
            detected_faces += 1
            face_images.append(cropped_face)
            face_image_file_name = name + "_face_" + image_real_number + ".jpg"
            face_images_files_names.append(face_image_file_name)
    
    if len(face_images) != 0: #Si au moins un visage a été détecté sur toutes les photos de la personne.
        path_face_folder = resize_path + name + "_FACES"
        os.mkdir(path_face_folder) #Création du dossier qui va contenir les photos de tous ses visages.
        for face_image_index in range(len(face_images)):
            path_to_write = path_face_folder + "/" + face_images_files_names[face_image_index]
            try:
                cv2.imwrite(path_to_write, cv2.cvtColor(face_images[face_image_index], cv2.COLOR_RGB2BGR)) #Création du fichier .jpg.
            except:
                pass
    message = str(detected_faces) + " faces detected on " + str(total_pictures) + " pictures."
    return message