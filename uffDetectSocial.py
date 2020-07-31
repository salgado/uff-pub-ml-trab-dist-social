############################
#  Deteccao de Distanciamento Social
#  Autor: Alex Salgado
# 
#  Para executar, usar o script start.sh
############################

import cv2 
import numpy as np
import uffBBoxYolo as mydet
import argparse
import time
import imutils
import testHomogr as uffHomg

# parametros de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="defina o caminho para o video")
args = vars(ap.parse_args())

classesFile = "cnn/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

#modelConfiguration = "cnn/yolov3-tiny.cfg"
#modelWeights = "cnn/yolov3-tiny.weights"
modelConfiguration = "cnn/yolov3-320.cfg"
modelWeights = "cnn/yolov3-320.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#parametros da deteccao
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
size_frame = 900

# inicializando video ou camera
print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(args["input"] if args["input"] else 0)

while True:
    prev_time = time.time()

    success, img = cap.read()

    # Resize na imagem
    img = imutils.resize(img, width=int(size_frame))
    #img = uffHomg.transform(img)			

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0]-1]) for i in net.getUnconnectedOutLayers()]
 
    outputs = net.forward(outputNames)

    # detecta distancia social
    mydet.findSocialDistance(outputs, img,confThreshold,nmsThreshold,classNames)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF

    # tecle `q` para sair
    if key == ord("q"):
        break
