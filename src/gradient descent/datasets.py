from numpy import *

    
class TwoDDiagonal:
    
    X = array([[5, 2, 5, 5],
        [5, 0, 5, 5],
        [5, 0, 4, 5],
        [5, 1, 5, 5],
        [5, 3, 4, 4],
        [5, 0, 5, 5],
        [5, 0, 5, 5],
        [4, 4, 4, 5],
        [2, 1, 3, 5],
        [5, 0, 5, 5],
        [4, 1, 5, 5],
        [5, 2, 4, 1],
        [5, 1, 5, 4],
        [1, 0, 5, 5],
        [5, 0, 5, 5],
        [3, 1, 5, 4],
        [5, 1, 5, 4],
        [5, 1, 5, 5],
        [2, 0, 5, 5],
        [3, 2, 5, 5],
        [4, 4, 5, 2],
        [5, 1, 4, 5],
        [3, 1, 5, 5],
        [4, 2, 5, 4],
        [2, 3, 5, 4],
        [2, 0, 5, 5],
        [5, 1, 5, 5],
        [4, 0, 5, 5],
        [2, 4, 5, 3],
        [5, 1, 4, 5],
        [3, 3, 5, 5],
        [5, 0, 5, 5],
        [4, 4, 5, 5],
        [5, 0, 5, 5],
        [5, 1, 4, 3],
        [4, 0, 5, 5],
        [5, 2, 5, 5],
        [5, 1, 5, 5],
        [2, 1, 5, 5],
        [5, 0, 5, 5],
        [2, 1, 5, 5],
        [3, 1, 5, 3],
        [5, 2, 5, 5],
        [2, 3, 5, 5],
        [5, 3, 5, 5],
        [5, 0, 5, 5],
        [2, 1, 5, 5],
        [5, 0, 4, 4],
        [2, 1, 5, 5],
        [3, 4, 5, 1],
        [5, 3, 5, 5],
        [4, 0, 5, 5],
        [2, 0, 5, 3],
        [2, 1, 3, 4],
        [3, 0, 5, 5],
        [5, 1, 4, 5],
        [5, 0, 5, 5],
        [5, 0, 5, 5],
        [2, 3, 5, 5],
        [2, 0, 5, 2],
        [2, 1, 5, 3],
        [4, 0, 5, 5],
        [3, 1, 5, 5],
        [5, 3, 3, 5],
        [4, 1, 5, 5],
        [5, 0, 5, 5],
        [2, 3, 5, 5],
        [5, 1, 5, 5],
        [5, 1, 5, 5],
        [3, 3, 5, 3],
        [5, 0, 5, 5],
        [5, 0, 5, 5],
        [2, 2, 5, 5],
        [3, 1, 5, 3],
        [5, 1, 5, 4],
        [5, 0, 5, 5],
        [2, 2, 5, 5],
        [1, 4, 5, 3],
        [3, 2, 5, 3],
        [5, 1, 5, 5],
        [4, 2, 2, 5],
        [1, 2, 5, 5],
        [2, 3, 5, 4],
        [3, 2, 5, 5],
        [2, 4, 5, 4],
        [3, 0, 5, 5],
        [3, 0, 5, 5],
        [5, 0, 4, 5],
        [5, 0, 5, 5],
        [5, 3, 3, 3],
        [5, 0, 5, 4],
        [5, 0, 5, 5],
        [5, 0, 5, 4],
        [2, 1, 5, 5],
        [2, 2, 5, 5],
        [3, 2, 5, 5],
        [2, 0, 5, 5],
        [5, 0, 5, 5],
        [5, 0, 5, 5],
        [2, 0, 5, 5]])
    
    Y = array([ 1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,
 -1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,
 -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
 -1, -1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, 1,  1, -1, -1, -1,
 -1,  1,  1, -1])
    
    Xte = X
    Yte = Y
