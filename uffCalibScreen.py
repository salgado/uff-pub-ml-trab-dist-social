import cv2
import numpy as np
import time

def gerarImagemVideo():
    # gera 1 imagem do video para calibracao manual
    


drawing = False
pontoIni = ()
pontoFim = ()
lista_pontos = []

def drawQuad():
    cv2.line(frame, pontoIni, pontoFim, (255,255,0), 1)
        
    i = 0
    print("Desenhar {pontoIni} - {pontoFim}")
    for ponto in lista_pontos:
        if i == 0:
            pIni = pFim = ponto
        else:
            pFim = ponto

        print(f"pontos {i} : {ponto}")
        print(f"pIni: {pIni}")
        print(f"pFim: {pFim}")
        cv2.line(frame, (pIni[0], pIni[1]), (pFim[0], pFim[1]), (0,255,0), 1)
        i = i + 1
        pIni = pFim
    

def mouse_drawing(event, x, y, flags, params):
    global lista_pontos, pontoFim, pontoIni, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        #drawing = True
        pontoIni = (x,y)
        lista_pontos.append([x,y])
    elif event == cv2.EVENT_MOUSEMOVE:
        #if drawing is True:
        pontoFim = (x, y)
    

#path = "/Users/alex/Downloads/video-pedestre-seq03-img-left/my.avi"

cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

while True:
    _, frame = cap.read()

    if pontoIni and pontoFim:
        drawQuad()

    cv2.imshow( "Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
