##edge detection&&facial -retain this as much as possible
##make image into triangle
##combine triangles up to a degree
##otsu only works for bimodal??
#point density


####pre-processing - denoising
import cv2
from EdgeDraw import EdgeDraw
import math
import numpy as np

def distance(p1, p2):
    dx = p1[0] -p2[0]
    dy = p1[1]-p2[1]
    return math.sqrt(dx*dx+dy*dy)


def saliency_sampling(img):
    '''

    :param img:
    :return: sampled point list
    '''
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#get binary map
    #TODO,random sampling N,default N = Floor(Lw/Li) × Floor(Lh/Li)

def get_edges(im,window =4):
    '''
    edge figures
    :return:
    '''
    edges = EdgeDraw().run(im,window)
    #edges = [np.array(e).reshape((len(e),1,2)) for e in edges]
    edges = np.array(edges).reshape(len(edges), 1,2)

    return edges

def mid_point(p1,p2):
    return (math.ceil((p1[0]+p2[0])/2),math.ceil((p1[1]+p2[1])/2))


def approximate_contour(raw, distance_constraint = 8.2, e_factor =0.00001 ):
    epsilon = e_factor * cv2.arcLength(raw, True)
    raw_approx = cv2.approxPolyDP(raw, epsilon, False)
  #  approx = []
  #  for idxP in range(len(raw_approx)-1):
  #      left = raw_approx[idxP][0]
  #      right = raw_approx[idxP + 1][0]
  #      if distance(left, right) > distance_constraint:
  #          mid = mid_point(left, right)
  #          approx.extend([left,mid])
  #      else:
  #          approx.append(left)
  #  approx.append(raw_approx[len(raw_approx)-1][0])
    return raw_approx





def edge_sampling(cnt):
    pass


def get_distance_map(d_map, m):
    #form distance map
    #https://github.com/mirrorworld/VoronoiJumpFlood/blob/master/jump.py
    return np.where(d_map/m%2==0, 255/m*(1-d_map%m),255/m*d_map%m)


def jump_flooding(edges, shape):
    #round up, 7 =>3
    distanceMap, featureMap = np.nan*np.empty(shape), np.nan*np.empty(shape)
    steps = math.ceil(math.log(max(shape)))
    #initiation
    for idx,c in enumerate(edges):
        x, y = c
        distanceMap[x,y] = 0
        featureMap[x,y] = idx
    for idxS in range(steps):
        window = pow(2, steps-idxS-1)
        jump_once(distanceMap, featureMap, window, shape)
    return distanceMap, featureMap


        #check 9 points
def jump_once(distanceMap, featureMap,  window, shape):
    for idxR in range(0, shape[0],window):
        for idxC in range(0, shape[1],window):
            if featureMap[idxR, idxC] == np.nan:#this is not related to any seed yet
                continue
            for idxRf in range(idxR-window, idxR, idxR+window):
                for idxCf in range(idxC-window, idxC, idxC+window):
                    d = distance((featureMap[idxR, idxC], (idxRf, idxCf)))
                    if d < distanceMap[idxRf, idxCf]:
                        distanceMap[idxRf, idxCf] = d
                        featureMap[idxRf, idxCf] = featureMap[idxR, idxC]#update nearest edge





##Utility function
def showImage( name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##Utility function
def edges2Image(plist, shape):
    pointFig = np.zeros(shape, np.uint8)
    for pointX, pointY in plist:
        pointFig[pointX, pointY] = [255, 255, 255]

    return pointFig


def helper(addr):
    im = cv2.imread(addr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    rawCnt = get_edges(im,2)
    #rawCnt = [(50,50),(50,100),(100,50),(100,100),(44,111),(51,11),(13,43)]
    #rawCnt = np.array(rawCnt).reshape((len(rawCnt),1,2))
    approCnt,raw_a = approximate_contour(rawCnt)
    showImage('contours', edges2Image(rawCnt.reshape(len(rawCnt),2), im.shape))
    showImage('contours', edges2Image(raw_a.reshape(len(raw_a),2), im.shape))
    showImage('contours', edges2Image(approCnt, im.shape))



####edge tracing sampling on edge
if __name__ =='__main__':
    helper('images/mouse.jpg')