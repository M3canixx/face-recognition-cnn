def main(person_ID, person_NAME, img_ID, face_IMG_file_name, dataset): #Ajoute un fichier et toutes ses informations au dataframe. (Insère une ligne)
    new_row = {
        'person_id': person_ID,
        'person_name': person_NAME,
        'img_id': img_ID,
        'face_img_file_name': face_IMG_file_name
        }
    dataset = dataset.append(new_row, ignore_index=True)
    return dataset