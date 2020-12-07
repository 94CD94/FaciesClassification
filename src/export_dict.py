import numpy as np


def exportfromkeys(Elenco, Richiesta, Indice): 
        Richiesta.reverse()
        Richiesta.reverse()
        Risultato=[]
        for i in Richiesta:
            for keys in Elenco:
                if i  == keys:
                    Risultato.append(Elenco[i][Indice])
        return(np.asarray(Risultato))
                
        
