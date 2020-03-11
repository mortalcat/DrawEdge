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

def get_edge_features(im):
    '''
    edge figures
    :return:
    '''
    return EdgeDraw(im).run()



def get_distance_map(im, edges):
    #form distance map
    #https://github.com/mirrorworld/VoronoiJumpFlood/blob/master/jump.py
    pass


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



def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1]-p2[1]
    return math.sqrt(dx*dx+dy*dy)








####edge tracing sampling on edge
