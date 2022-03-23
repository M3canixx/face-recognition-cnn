import matplotlib.pyplot as plt

def main(df): #Entrée: dataframe et chemin du dossier data. Renvoie X (les images) et y (les labels).
    IDs = list(df['person_id'])
    FILEs = list(df['face_img_file_name'])
    number_of_images = len(IDs)

    print(IDs)
    print(FILEs)
    print(number_of_images)
    
    X = []
    y = []
    
    for index in range(number_of_images):
        ID = int(IDs[index])
        file_path = FILEs[index]
        image = plt.imread(file_path)
        X.append(image)
        y.append(ID)
    return X, y