import cv2 
import numpy as np
from itertools import combinations
import math
import uffConstant as constant
from scipy.spatial import distance as dist

# verifica se a distancia social foi violada
def is_distance_violated(point1, point2, img):
    #(x,y,w,h)
    pe1 = np.array([point1[0]+point1[2]/2, point1[1]+point1[3]])
    pe2 = np.array([point2[0]+point2[2]/2, point2[1]+point2[3]])
    
    # desenha referencia
    cv2.circle(img, (int(pe2[0]),int(pe2[1])), 5,(0,0,222), -1) 
    cv2.circle(img, (int(pe1[0]),int(pe1[1])), 5,(0,0,222), -1) 
    
    if calcDistance(pe1, pe2) < constant.DIST_SOCIAL :
        return True
    else:
        return False

def calcDistance(point1, point2):
    # compute the Euclidean distance between the midpoints
    H, H_INV = constant.transform()

    print(f"Matrixxxxx-{H} ")
    newp = np.matmul( np.float32([point1[0], point1[1],1]), H);
    newp2 =  np.float32([point2[0], point2[1],1]) @ H;
    print(f"nem point : {newp}")
    print(f"nem point : {newp2}")

    #p1 = cv2.warpPerspective()

    dA = dist.euclidean((point1[0], point1[1]), (point2[0], point2[1]))
    print(f"distancia EUCLIDIANA ======= ({point1} , {point2}) = {dA}")
    
    #nova coordenada
    #dA = dist.euclidean((newp[0], newp[1]), (newp2[0], newp2[1]))
    #print(f"distancia 222 -EUCLIDIANA ======= ({newp} , {newp2}) = {dA}")
    
    return dA


def is_close(p1, p2):
    """
    #================================================================
    # 1. Calcular distancia euclidiana
    #================================================================    
    """
    dst = math.sqrt(p1**2 + p2**2)
    print(f"distancia ======= {dst}")
    
    return dst 


def findSocialDistance(outputs,img, confThreshold, nmsThreshold, classNames):

    # filtra por classe person - pessoa/pedestre
    if len(outputs) > 0: 
        
        hT, wT, cT = img.shape
        bbox = []      # bounding box corner points
        classIds = []  # class id with the highest confidence 
        confs = []     # confidence value of the highest class
        for output in outputs:
        
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                name_class = classNames[classId].upper()
                if confidence > confThreshold and name_class == 'PERSON':
                    #converte as coordenadas
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
                    
        
        centroid_dict = dict() 
        objectId = 0

        # aplicar non-maxima supression para evitar bbox duplicados
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
        
            # cria lista com centro da pessoa detectada.
            centroid_dict[objectId] = (x,y,w,h)
            objectId += 1  

            
        # Passo 2 : Determinar qual bbox da pessoa esta prxima da outra
        red_zone_list = [] 
        red_line_list = []
        print(f"inicio detector centroid_dict : {centroid_dict}")
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # combinacao entre dois pontos p1 e p2
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	
            
            # Calcula distancia
            distance_violated = is_distance_violated(p1, p2, img) 			# Calculates the Euclidean distance
            print(f"distancia detector:{distance_violated}")

            if distance_violated :						
                #cv2.circle(img, (p1[0],p1[1]), 5,(0,0,222), -1)
                #cv2.circle(img, (p2[0],p2[1]), 5,(0,0,222), -1)
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  adiciona lista de violadores
                    red_line_list.append(p1[0:2])   #  da distancia
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		
                    red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  
            x,y,w,h = box[0], box[1], box[2], box[3] 
            if idx in red_zone_list:
                cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0 , 255), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255 , 0), 2)
 