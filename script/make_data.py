import pandas as pd
import os
import addImage
def main(folder_path, person_id_make_data, name, data): #Crée le dataframe

    dataframe = pd.DataFrame(data)
    person_id_link = []
    
    persons = os.listdir(folder_path)
    
    images_files_names = os.listdir(folder_path)
    
    id_name_tuple = (person_id_make_data, name)
    person_id_link.append(id_name_tuple)
    
    for image_index in range(len(images_files_names)):
        img_file_name = images_files_names[image_index]
        image_id = int(img_file_name.split('_')[-1].split('.')[0])
        face_image_file_name = folder_path + "/" + persons[image_index]
        
        dataframe = addImage.main(
            person_ID = person_id_make_data,
            person_NAME = name,
            img_ID = image_id,
            face_IMG_file_name = face_image_file_name,
            dataset = dataframe
        )
    person_id_make_data += 1
    dataframe = dataframe.astype({'person_id': 'int32'})
    dataframe = dataframe.astype({'img_id': 'int32'})
    dataframe.to_csv("data/df_faces.csv", index=False, header=True)
    return dataframe, person_id_make_data