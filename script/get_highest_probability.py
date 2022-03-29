#Revoie l'index du tableau avec la valeur la plus élevée.
def main(tab):
    maximum=0
    index = 0
    for i in range(len(tab)):
        if maximum < tab[i]:
            maximum = tab[i]
            index = i
    return index