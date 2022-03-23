import find_minimum
import delete_case
def main(array1, array2) : #Trie array2 selon array1 dans l'ordre croissant
    newArray1 = []
    newArray2 = []
    l = len(array1)
    for i in range(l) :
        posMinimum = find_minimum.main(array1)
        newArray1 = newArray1 + [array1[posMinimum]]
        newArray2 = newArray2 + [array2[posMinimum]]
        array1 = delete_case.main(array1, posMinimum)
        array2 = delete_case.main(array2, posMinimum)
    return newArray1, newArray2