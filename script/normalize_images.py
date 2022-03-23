def main(list_of_images): #Normalise les images
    l = len(list_of_images)
    for i in range(l):
        list_of_images[i] = list_of_images[i]/255.
    return list_of_images