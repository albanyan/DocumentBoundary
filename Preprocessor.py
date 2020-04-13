import numpy as np
import cv2 as cv
import fitz
from PIL import Image
from config import ratio,dim

class Preprocessor():
    def __init__(self, PDF_Path):
        self.PDF_Path = PDF_Path
        self.imgs = self.PDF2img()
    def CropIt(self,img,ratio):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return img[int(img.shape[0]*(1-ratio)):-1]
    def PDF2img(self):
        doc = fitz.open(self.PDF_Path)
        imgs = []
        for p in range(doc.pageCount):
            page = doc.loadPage(p) #number of page
            pix = page.getPixmap()
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            imgs.append(np.asarray(img))
        doc.close()
        return imgs
    def Preprocess(self):
        for i in range(len(self.imgs)):
            img=self.CropIt(self.imgs[i],ratio)
            img = cv.bitwise_not(img)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            img=img/255
            self.imgs[i] = img
        return self.imgs
