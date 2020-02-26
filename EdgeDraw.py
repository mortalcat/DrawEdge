import cv2
import numpy as np
##Implementation of EdgeDrawing Algorithm
##Input a image, output its edge
class EdgeDraw(object):
    def __init__(self, addrIn, addrOut):
        self.addrIn = addrIn
        self.addrOut = addrOut


    @staticmethod
    def smooth(img):
        ##non-local denoising, patch window 7
        noiselessImage = cv2.fastNlMeansDenoisingColored(
            img, None, 10, 10, 7, 21)
        return noiselessImage

    @staticmethod
    def cutEdge(img):
        rs,cs = img.shape
        img [0, :] = 0
        img [rs-1,:] = 0
        img [:, 0] = 0
        img [:, rs -1] = 0

    @staticmethod
    def findAnchors(img, kernel = 5, thre = 8):
        ###get a edge gradience and direction map
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        magMap =  np.sqrt(dx*dx + dy*dy)
        EdgeDraw.cutEdge(magMap)
        direMap = np.where(np.abs(dx) > np.abs(dy), np.ones(dx.shape),np.zeros(dx.shape))#1 for horizontal
        y_pre_1 = np.roll(magMap, -1, axis=0)
        y_next_1 = np.roll(magMap, 1, axis=0)
        x_pre_1 = np.roll(magMap, -1, axis=1)
        x_next_1 =np.roll(magMap, 1, axis=1)
        ##todo: waht to do with edges, probably just wipe all them to 0?
        t = np.where(((direMap == 1) & (np.abs(magMap - y_next_1) > thre) & (np.abs(magMap - y_pre_1) > thre)) | ((direMap == 0) & (np.abs(magMap - x_pre_1) > thre) & (np.abs(magMap - x_next_1) > thre))) #isAnchor, dab 1
        anchors = list(zip(*t))
        print(anchors)
        return magMap,direMap,anchors#x, y coordinates of anchors


    @staticmethod
    def drawEdges(magMap, direMap, anchors):
        '''
        smart route all edges from anchors
        :param anchorMap
        :return:
        '''
        #for each point, check three neighbours depend on direction
        def moveOnce(direction, pX, pY):
            if direction is 'left':
                neighbours= magMap[pX-1,pY-1:pY+2]
                ind = np.argmax(neighbours, axis=None)
                return (pX-1, pY+ind-1)
            elif direction is 'right':
                neighbours = magMap[pX+1, pY-1:pY+2]
                ind = np.argmax(neighbours, axis=None)
                return (pX+1, pY+ind-1)
            elif direction is 'up':
                neighbours = magMap[idxX-1:idxX+2, idxY - 1]
                ind = np.argmax(neighbours, axis=None)
                return (pX+ind-1, pY-1)
            else:
                neighbours = magMap[idxX-1:idxX+2, idxY + 1]
                ind = np.argmax(neighbours, axis=None)
                return (pX+ind-1, pY+1)
            return newPoint

        def moveTillEnd(direction, idxX, idxY):
            edge = [ (idxX,idxY)]
            stopFlag = False
            while magMap[idxX][idxY] > 0 and not stopFlag:
                newPoint = moveOnce(direction, idxX, idxY)
                if newPoint in anchors:
                    stopFlag = True
                else:
                    edge.append(newPoint)
                    idxX, idxY = newPoint

            return edge

        edges=[]
        def addEdge(edge):
            if len(edge)>1:
                edges.append(edge)
        #todo:handle connectivity by excluding duplicate points and concatenate?
        for idxX,idxY in anchors:
            if direMap[idxX][idxY] == 1:  # Horizontal
                el = moveTillEnd('left', idxX, idxY)
                er = moveTillEnd('right', idxX, idxY)
                addEdge(el)
                addEdge(er)
            if direMap[idxX][idxY] == 0:  # Vertical
                eU = moveTillEnd('up', idxX, idxY)
                eD = moveTillEnd('down', idxX, idxY)
                addEdge(eU)
                addEdge(eD)
        return edges
    @staticmethod
    def showEdges(edges,imshape):
        pointFig = np.zeros(imshape, np.uint8)
        for edge in edges:
            for pointX, pointY in edge:
                pointFig[pointX, pointY ] = (0, 0, 0)

        EdgeDraw.showImage('edges', pointFig)

    @staticmethod
    def showImage( name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def run(self):
        '''
        main function
        :return:
        '''
        ###read image
        im = cv2.imread(self.addrIn)
        ###denoising
        noiseless = self.__class__.smooth(im)
        self.__class__.showImage('smooth over', noiseless)
        ###gray
        gray = cv2.cvtColor(noiseless, cv2.COLOR_BGR2GRAY)
        ###find anchors
        ###smart rounting
        magMap, direMap, anchors = self.__class__.findAnchors(gray)
        edges = self.__class__.drawEdges(magMap, direMap, anchors)
        self.__class__.showEdges(edges, im.shape)

if __name__ == '__main__':
    e = EdgeDraw('./images/section8-image.png', './images/edges.jpg')
    e.run()



