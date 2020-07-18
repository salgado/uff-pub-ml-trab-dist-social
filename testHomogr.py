import cv2 as cv
import numpy as np
import imutils
import time


def pixelPerMetric():
    #Quantos pixel tem em 1 metro
    #pixels_per_metric = 150px / 0.955in = 157px
    #pixels_per_metric = 15.25px / 1.0m = 15.25px
    #largura:18.5, altura:12.0

    pixel_per_metric = 15.25 / 1.0
    return pixel_per_metric

def refLargAlt(pts):
    w = ((pts[0][0]-pts[2][0]) + (pts[1][0]-pts[3][0]))/2
    h = ((pts[0][1]-pts[1][1]) + (pts[2][1]-pts[3][1]))/2
    return w, h

def transform(frame):

    w, h = 400,400
    pts1 = np.float32([[366,498],[746,98],[75,389],[567,79]])
    pts1 = np.float32([[539,502],[815,103],[8,250],[411,69]])
    pts2 = np.float32([[w,h],[w,0],[0,h],[0,0]])

    matrix = cv.getPerspectiveTransform(pts1,pts2)
    h,w = frame.shape[:2]
    output = cv.warpPerspective(frame, matrix,(w,h))
    
    # pontos referencia
    ptsRef = np.float32([[285,277],[285,265],[266,279],[267,267]])
    wref, href = refLargAlt(ptsRef)
    #print(f"largura:{wref}, altura:{href}")
    #largura:18.5, altura:12.0
    
    for x in range(0,4):
        cv.circle(frame, (pts1[x][0], pts1[x][1]),5,(0,0,255),cv.FILLED) 
        cv.circle(output, (ptsRef[x][0], ptsRef[x][1]),1,(255,0,0),cv.FILLED) 
    
    #cv.imshow("tela", frame)
    #cv.imshow("bird", output)
    return output
    

def testHomography():
    path="videos/pedestrians.mp4"

    cap = cv.VideoCapture(path)

    size_frame = 900

    while True:    
        # Load the frame and test if it has reache the end of the video
        (frame_exists, frame) = cap.read()
        frame = imutils.resize(frame, width=int(size_frame))
        
        transform(frame)

        key = cv.waitKey(1)
        if key == 27:
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
