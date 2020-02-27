import cv2
import numpy as np
##Implementation of EdgeDrawing Algorithm
##Input a image, output its edge
class EdgeDraw(object):
    def __init__(self, addrIn, addrOut):
        self.addrIn = addrIn
        self.addrOut = addrOut


    def smooth(self,img):
        img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    def cutEdge(self, img):
        rs,cs = img.shape
        img [0, :] = 0
        img [rs-1,:] = 0
        img [:, 0] = 0
        img [:, cs -1] = 0

    def getThreshould(self, img):
        ##get threshould of the
        high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return high_thresh


    def findAnchors(self, img, window = 4, g_thre = 36,anchorthre = 8):
        ###get a edge gradience and direction map
        otsuthre = self.getThreshould(img)
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        magMap =  np.sqrt(dx*dx + dy*dy)
        self.cutEdge(magMap)
        maxV = np.max(magMap)
        g_thre = otsuthre*0.25/255 * maxV#normalization
        magMap = np.where(magMap >= g_thre, magMap, 0)
        print('thre: ', anchorthre)
        direMap = np.where(np.abs(dy) > np.abs(dx), np.ones(dx.shape),np.zeros(dx.shape))#1 for horizontal
        y_pre_1 = np.roll(magMap, -1, axis=0)
        y_next_1 = np.roll(magMap, 1, axis=0)
        x_pre_1 = np.roll(magMap, -1, axis=1)
        x_next_1 =np.roll(magMap, 1, axis=1)
        anchorMap = np.where(((direMap == 1) & (magMap - y_next_1 > anchorthre) & (magMap - y_pre_1 > anchorthre)| ((direMap == 0) & (magMap - x_pre_1 > anchorthre) & (magMap - x_next_1 > anchorthre))), np.ones(magMap.shape),np.zeros(magMap.shape))
        ####window selection######################################
        rsize, csize = magMap.shape
        copyMap = np.zeros(magMap.shape)
        for ir in range(0,rsize,window):
            for ic in range(0,csize,window):
                copyMap[ir,ic] = anchorMap[ir,ic]
        #ranchors = list(zip(*np.where(anchorMap>0)))
        #EdgeDraw.showEdges(ranchors,[img.shape[0],img.shape[1],3])
        #print(ranchors)
        anchors = list(zip(*np.where(copyMap > 0)))
        #self.showEdges(anchors,[img.shape[0],img.shape[1],3])
        return magMap,direMap,anchors#x, y coordinates of anchors


    def drawEdges(self,magMap, direMap, anchors):
        edges=[]
        #for each point, check three neighbours depend on direction
        def moveOnce(direction, py, px):
            if direction is 'up':
                neighbours= magMap[py - 1, px - 1:px + 2]
                ind = np.argmax(neighbours, axis=None)
                return (py - 1, px + ind - 1)
            elif direction is 'down':
                neighbours = magMap[py + 1, px - 1:px + 2]
                ind = np.argmax(neighbours, axis=None)
                return (py + 1, px + ind - 1)
            elif direction is 'left':
                neighbours = magMap[py-1:py+2, px - 1]
                ind = np.argmax(neighbours, axis=None)
                return (py + ind - 1, px - 1)
            elif direction is 'right':
                neighbours = magMap[py-1:py+2, px + 1]
                ind = np.argmax(neighbours, axis=None)
                return (py + ind - 1, px + 1)

        def moveTillEnd(ind, idx_y, idx_x):
            stopFlag = False
            dire = ['left','right','up','down']
            while magMap[idx_y][idx_x] > 0 and not stopFlag:
                newPoint = moveOnce(dire[ind], idx_y, idx_x)
                if newPoint in anchors or newPoint in edges:
                    stopFlag = True
                else:
                    edges.append(newPoint)
                    idx_y, idx_x = newPoint

        #todo:handle connectivity by excluding duplicate points and concatenate?
        for idxX,idxY in anchors:
            if direMap[idxX][idxY] == 1:
                moveTillEnd(0,idxX, idxY)
                moveTillEnd(1,idxX, idxY)
            else:
                moveTillEnd(2, idxX, idxY)
                moveTillEnd(3, idxX, idxY)

        return edges

    def showEdges(self, edges,imshape):
        pointFig = np.zeros(imshape, np.uint8)
        for pointX, pointY in edges:
            pointFig[pointX, pointY] = [255, 255, 255]

        return pointFig

    @staticmethod
    def showImage( name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def run(self, show = False):
        '''
        main function
        :return:
        '''
        ###read image
        im = cv2.imread(self.addrIn)
        ###denoising
        noiseless = self.smooth(im)
        if show:
            self.showImage('smooth over', noiseless)
        ###gray
        gray = cv2.cvtColor(noiseless, cv2.COLOR_BGR2GRAY)
        ###canny for comparision
        otsuthre = self.getThreshould(gray)
        canny = cv2.Canny(gray, 0.5*otsuthre, otsuthre)
        #cv2.imwrite('./images/gray.jpg', gray)
        cv2.imwrite('./images/canny.jpg', canny)
        ###find anchors
        ###smart rounting
        magMap, direMap, anchors = self.findAnchors(gray)
        edges = self.drawEdges(magMap, direMap, anchors)
        out = self.showEdges(edges, im.shape)
        if show:
            self.showImage('edges', out)
        cv2.imwrite('./images/edges.jpg', out)


if __name__ == '__main__':
    e = EdgeDraw('./images/dog.jpg', './images/edges.jpg')
    e.run()