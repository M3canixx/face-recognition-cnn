import cv2
def main(image): #Renvoie l'image d'entrée en niveau de gris
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)