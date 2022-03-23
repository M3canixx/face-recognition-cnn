def main(array,indice) : #Supprime la case d'un tableau selon sa position
    newArray = []
    l = len(array)
    for i in range(l) :
        if indice != i :
            newArray = newArray + [array[i]]
    return newArray