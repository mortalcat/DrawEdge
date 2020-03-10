##edge detection&&facial -retain this as much as possible
##make image into triangle
##combine triangles up to a degree
##otsu only works for bimodal??
#point density


####pre-processing - denoising
import cv2
import EdgeDraw
def saliencySampling(img):
    '''

    :param img:
    :return: sampled point list
    '''
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#get binary map
    #TODO,random sampling N,default N = Floor(Lw/Li) × Floor(Lh/Li)

def getEdgeFeatures(im):
    '''
    edge figures
    :return:
    '''
    return EdgeDraw(im).run()



def getDistanceMap(im, edges):
    #form distance map
    #https://github.com/mirrorworld/VoronoiJumpFlood/blob/master/jump.py
    pass




####edge tracing sampling on edge
