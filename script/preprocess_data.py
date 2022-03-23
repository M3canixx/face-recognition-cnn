import gray_image
import normalize_images
import numpy as np
from keras.utils.np_utils import to_categorical

def main(X, y, gray): #Rend les données utilisables par le modèle. gray ---> booléen, True si on veut les images en niveau de gris, False pour RGB.
    if gray:
        for i in range(len(X)):
            X[i] = gray_image.main(X[i])
        X = np.asarray(X)
        X = X/255.
        X = X.reshape(X.shape[0], 224, 224, 1)
    else:
        X = normalize_images.main(X)
        X = np.asarray(X)
        X = X.reshape(X.shape[0], 224, 224, 3)
    y = to_categorical(y)
    return X, y