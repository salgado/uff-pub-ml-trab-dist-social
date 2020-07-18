import Algorithmia
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import json


def mostraLinhaCartesiano():

    path = "/Users/alex/Desktop/pedestre.png"
    
    # Using cv2.imread() method 
    im = cv2.imread(path) 
    
    # https://data.vision.ee.ethz.ch/cvl/aess/dataset/
    left, right = [-720, 389.72832844401313],[720, 432.6942735506478] # cenarua
    left, right = [-280, 4.309184419248052], [280, 3.306936588562465] # casaverde
    left, right = [-227.5, 3.3713737513344366], [227.5, 2.5570473889023955] # casaparati
    left, right = [-480, 156.3165168998381], [480, 161.58070852259667] # abbey
    left, right = [-320, 9.507465923034069], [320, 18.224261591858593]
    sz = im.shape

    plt.figure(1)
    plt.imshow(im, extent=[-sz[1]/2, sz[1]/2, -sz[0]/2, sz[0]/2])
    plt.plot([left[0], right[0]], [left[1], right[1]], 'r')
    ax = plt.gca();
    ax.autoscale_view('tight')
    plt.show()

def gravaHorizonte():
    return True

def hLW():  
    # path 
    path = "1280px-HFX_Airport_4.jpg"
    
    # Using cv2.imread() method 
    img = cv2.imread(path) 

    input = {
    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/HFX_Airport_4.jpg/1280px-HFX_Airport_4.jpg"
    }
    
    # {'left': [-716.5, 333.3337877599805], 'right': [716.5, 376.0908706334579]}

    imagem2 = {
        "image": "https://raw.githubusercontent.com/salgado/publictemp/master/pedestre.png"
        #"image": "https://www.earthcam.com/share/temp_images/1594654447140.jpg"
    }
    
    client = Algorithmia.client('simzQM8BgYKMpiFvKnN13YJs+Oj1')
    algo = client.algo('ukyvision/deephorizon/0.1.0')
    algo.set_options(timeout=300) # optional
    print(algo.pipe(imagem2).result)


def transform2im(p1, w, h):
  print(f"[w/2 + p1[0]:{w/2} , {p1[0]}")  
  new_p1 = np.float32( [w/2 + p1[0], h/2 - p1[1]])
  return new_p1

def plotAffine(reta, img):
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    #H = np.array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
    #im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))

    cols,rows = img.shape[:2]

    M = cv2.getAffineTransform(pts2,pts1)
    H = np.array([[  0.68181818 , -0.40909091 , 84.09090909],
 [  0.22727273,   0.86363636, -38.63636364]])
    
    # equacao 2.19 do livro Hartley, Zisserman a ser ajustado no codigo
    H = np.array([[  1.0 , 0.0 , 0.0],
                [  0.0,   1.0, 0.0],
                [  -5.45811866e-04, -1.82928395e-02,  1.00000000e+00]])
    newH = np.matmul( np.array([0,0,1]).T, H )   
    print(f"Matriz H :{M}")

    #dst = cv2.warpAffine(img,H,(cols,rows))
    dst = cv2.warpPerspective(img,newH,(rows,cols))



    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def showOrigHorizon():
    color = (0, 255, 0) 
    
    # Line thickness of 9 px 

    path = "people.png"
    path = "/Users/alex/Desktop/cenarua.png"
    
    # Using cv2.imread() method 
    img = cv2.imread(path) 
    
    left, right = [-716.5, 333.3337877599805],[716.5, 376.0908706334579]
    left, right = [-720, 389.72832844401313],[720, 432.6942735506478]
    
    #w = 776
    #h = 1433
    h,w = img.shape[:2]

    p1 = np.float32( left)
    p2 = np.float32( right )

    print (f"p1:{p1}")
    p1 = transform2im (p1, w, h)
    print (f"p1:{p1}")

    print (f"p2:{p2}")
    p2 = transform2im (p2, w, h)
    print (f"p2:{p2}")

    print(f"shape:{img.shape}")

    # compute line
    reta = np.cross([p1[0],p1[1],1],[p2[0],p2[1],1])
    #reta = np.cross([p1[0],p1[1],1],[p2[0],p2[1],1])
    #reta = np.cross(p2,p1)
    print (f"reta : {reta}" )
    reta /= reta[2] 
    print (f"reta : {reta}" )

    thickness = 6
    sz = img.shape
    # Using cv2.line() method 
    # Draw a diagonal green line with thickness of 9 px 
    img = cv2.circle(img, (p1[0], p1[1]), 50, (255,0,0), thickness) 
    img = cv2.circle(img, (p2[0], p2[1]), 50, (0,0,0), thickness) 
    img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness) 

    imS = cv2.resize(img, (w//1, h//1))                    # Resize image

    #plotAffine(reta, img)

    cv2.imshow('image',imS )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def showHorizon():
    #{'left': [-640, 294.98985200147615], 'right': [640, 292.69899981705186]}

    start_point = (-640, 295) 
    
    # End coordinate, here (250, 250) 
    # represents the bottom right corner of image 
    end_point = (640, 293) 
    
    # Green color in BGR 
    color = (0, 255, 0) 
    
    # Line thickness of 9 px 

    path = "people.png"
    
    # Using cv2.imread() method 
    im = cv2.imread(path) 
    

    
    thickness = 6
    sz = im.shape
    left, right = [-716.5, 333.3337877599805],[716.5, 376.0908706334579]
    # Using cv2.line() method 
    # Draw a diagonal green line with thickness of 9 px 
    #img = cv2.circle(img, start_point, 5, (255,0,0), thickness) 
    #img = cv2.circle(img, end_point, 5, (0,0,0), thickness) 
    #img = cv2.line(img, start_point, end_point, color, thickness) 

    p1 = np.float32( left)
    p2 = np.float32( right )

    #w = 776
    #h = 1433
    w,h = im.shape[:2]

    print (f"p1:{p1}")
    p1 = transform2im (p1, w, h)
    print (f"p1:{p1}")

    print (f"p2:{p2}")
    p2 = transform2im (p2, w, h)
    print (f"p2:{p2}")

    print(f"shape:{im.shape}")

    plt.figure(1)
    plt.imshow(im, extent=[-sz[1]/2, sz[1]/2, -sz[0]/2, sz[0]/2])
    plt.plot([left[0], right[0]], [left[1], right[1]], 'r')
    ax = plt.gca();
    ax.autoscale_view('tight')
    plt.show()

    """  
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
def affImg():
    start_point = np.array([-640, 295]) 
    end_point = np.array([640, 293]) 

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    path = "1280px-HFX_Airport_4.jpg"
    path = "people.png"
    
    # Using cv2.imread() method 
    img = cv2.imread(path) 
    rows,cols,ch = img.shape
    
    H = np.array([[1,0,0],[0,1,0],[  2/376320,    1280/376320, -376320/376320]])
    H = np.array([[1,0,0],[0,1,0],[ 0.00008464591923007647, -0.002820874471086036, 1]])

    h, status = cv2.findHomography(pts1, pts2)
    print(f"h:shape{h.shape} , h:{h}" )
    im_dst = cv2.warpPerspective(img, H,(cols, rows))

    cv2.imshow( "Destino",im_dst)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(im_dst),plt.title('Output')
    plt.show()

#hLW()
#affImg()
#showHorizon()
#showOrigHorizon()
mostraLinhaCartesiano()