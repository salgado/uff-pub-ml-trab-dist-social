import numpy as np
import cv2 as cv

DIST_SOCIAL = 40.5   #calculado por uffCalibScreen 2 metros
MATRIX = None

# testes para tirar perpectiva da cena manualmente devido ao erro do HLW
def transform():  
    w, h = 400,400
    pts1 = np.float32([[366,498],[746,98],[75,389],[567,79]])
    pts1 = np.float32([[539,502],[815,103],[8,250],[411,69]])
    pts2 = np.float32([[w,h],[w,0],[0,h],[0,0]])

    MATRIX = matrix = cv.getPerspectiveTransform(pts1,pts2)
    matrix_INV = cv.getPerspectiveTransform(pts2,pts1)
    return matrix, matrix_INV