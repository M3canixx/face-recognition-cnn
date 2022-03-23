def main(array) : #Renvoie la position de la plus petite valeur dans un tableau
    l = len(array)
    position = 0
    minimum = array[0]
    for i in range(1, l) :
        if array[i] < minimum :
            position = i
            minimum = array[i]
    return position