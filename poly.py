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


def saliency_sampling(img,sample_num,saliency_factor=0.7):
    '''
    :param img:
    :return: sampled point list
    '''
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    showImage('saliency',saliencyMap)
    saliencyMap = saliencyMap*255
    saliencyMap =saliencyMap.astype("uint8")
    showImage('saliency',saliencyMap)
    threshMap = cv2.threshold(saliencyMap, 0, 255,
                              cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]#get binary map
    print(success)
    showImage('binarySaliency',threshMap)
    front_num = saliency_factor * sample_num
    bg_num = (1-saliency_factor) * sample_num
    width, height = img.shape
    front_samples = []
    back_samples = []
    #TODO,random sampling N,default N = Floor(Lw/Li) × Floor(Lh/Li)
    while len(front_samples) < front_num and len(back_samples) <bg_num:
        px = np.random.randint(low=0, high=width)
        py = np.random.randint(low=0, high=height)
        if threshMap[py,px] == 1:#white, is front
            front_samples.append((py ,px))
        else:
            back_samples.append((py, px))
    return front_samples+back_samples


def get_edges(im,window =4):
    '''
    edge figures
    :return:
    '''
    edges = EdgeDraw().run(im,window)
    edges = [np.array(e).reshape((len(e),1,2)) for e in edges]
    #edges = np.array(edges).reshape(len(edges), 1,2)

    return edges

def mid_point(p1,p2):
    return (math.ceil((p1[0]+p2[0])/2),math.ceil((p1[1]+p2[1])/2))

#TODO:minimum length
#TODO:add 4 corner points
def approximate_contour(edges, distance_constraint = 8.2, e_factor =0.00001):
    #epsilon = e_factor * cv2.arcLength(raw, True)
    raw_approxs = [cv2.approxPolyDP(e, e_factor * cv2.arcLength(e, True), False) for e in edges]
    approxs =[]
    for raw_a in raw_approxs:
        approx = []
        for idxP in range(len(raw_a)-1):
            left = raw_a[idxP][0]
            right = raw_a[idxP + 1][0]
            if distance(left, right) > distance_constraint:
                mid = mid_point(left, right)
                approx.extend([left,mid])
            else:
                approx.append(left)
        approx.append(raw_a[len(raw_a)-1][0])
        approxs.append(approx)
    return approxs,raw_approxs

def get_feature_flow(edges, shape, m):
    #form distance map
    #https://github.com/mirrorworld/VoronoiJumpFlood/blob/master/jump.py
    d_map, f_map = jump_flooding(edges,shape)
    showImage('dmap',d_map.astype("uint8"))
    return np.where(d_map/m%2==0,255/m*d_map%m,255/m*(1-d_map%m))
def jump_flooding(edges, shape):
    #round up, 7 =>3
    distanceMap = 100000*np.ones(shape)
    featureMap = np.nan*np.empty(shape)
    steps = math.ceil(math.log(max(shape),2))
    #initiation
    for idx,c in enumerate(edges2One(edges)):
        x, y = c
        distanceMap[x][y] = 0
        featureMap[x][y] = idx
    for idxS in range(steps):
        window = pow(2, steps-idxS-1)
        jump_once(edges2One(edges),distanceMap, featureMap, window, shape)
    return distanceMap, featureMap

        #check 9 points
def jump_once(edges, distanceMap, featureMap, window, shape):
    for idxR in range(shape[0]):
        for idxC in range(shape[1]):
            if np.isnan(featureMap[idxR, idxC]):#this is not related to any seed yet
                continue
            for idxRf in [max(0,idxR-window), idxR, min(idxR+window, shape[0]-1)]:
                for idxCf in [max(0, idxC - window), idxC, min(idxC + window, shape[1]-1)]:
                    idxEdge = int(featureMap[idxR, idxC])
                    d = distance(edges[idxEdge], (idxRf, idxCf))
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

def edges2One(edges):
    one = []
    for e in edges:
        if type(e) == list:
            one.extend(e)
        else:
            one.extend(e.reshape(len(e),2))
    return one

def helper(addr):
    im = cv2.imread(addr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    edges = get_edges(im,2)
    sample_window = (im.shape[0]+im.shape[1])*0.02
    #rawCnt = [(50,50),(50,100),(100,50),(100,100),(44,111),(51,11),(13,43)]
    #rawCnt = np.array(rawCnt).reshape((len(rawCnt),1,2))
    approCnt,raw_a = approximate_contour(edges)
    showImage('edges', edges2Image(edges2One(edges), im.shape))
    showImage('Approximate breakdown', edges2Image(edges2One(approCnt), im.shape))
    feature_map = get_feature_flow(edges, gray.shape,sample_window/2)
    showImage('feature map', feature_map.astype("uint8"))



####edge tracing sampling on edge
if __name__ =='__main__':
    helper('images/taichi.jpg')
