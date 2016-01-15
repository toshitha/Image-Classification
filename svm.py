import cv2
import os 
from os.path import abspath, join
import numpy as np

bin_n = 16 # Number of bins

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

trainingImages = []
trainingLabels = []

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC)

for f in os.listdir('Train/Face/'):
    if f.endswith(".JPEG"):
        filename = join(abspath('Train/Face/'),f)
        image = cv2.imread(filename,0)
        blur = cv2.blur(image,(5,5))
        #laplacian = cv2.Laplacian(image,cv2.CV_64F)
        edge = cv2.Canny(blur,100,200)
        laplacian = cv2.Laplacian(blur,cv2.CV_64F)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
        hogF = hog(blur)
        #imhog = hog.compute(image)
        
        img = np.array(laplacian, dtype = np.float32).flatten()
        trainingImages.append(img)
        trainingLabels.append(1)


for f in os.listdir('Train/Not_Face/'):
    if f.endswith(".JPEG"):
        filename = join(abspath('Train/Not_Face/'),f)
        image = cv2.imread(filename,0)
        blur = cv2.blur(image,(5,5))
        laplacian = cv2.Laplacian(blur,cv2.CV_64F)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
        edge = cv2.Canny(blur,100,200)
        hogF = hog(blur)
        #imhog = hog.compute(image)
        img = np.array(laplacian, dtype = np.float32).flatten()

        trainingImages.append(img)
        trainingLabels.append(-1)

svm = cv2.SVM()
svm.train_auto(np.asarray(trainingImages),np.asarray(trainingLabels), None, None, params=svm_params, k_fold = 3)
#svm.train(np.asarray(trainingImages),np.asarray(trainingLabels), params=svm_params)
svm.save('svm_data.dat')

######################### Test ########################################################################

face = 0.0;
not_face = 0.0;

for f in os.listdir('Test/Face/'):
    if f.endswith(".JPEG"):
        filename = join(abspath('Test/Face/'),f)
        image = cv2.imread(filename,0)
        blur = cv2.blur(image,(5,5))
        laplacian = cv2.Laplacian(blur,cv2.CV_64F)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
        edge = cv2.Canny(blur,100,200)
        hogF = hog(blur)
        #imhog = hog.compute(image)
        img = np.array(laplacian, dtype = np.float32).flatten()

        result = svm.predict(img)

        if result == 1:
            face = face + 1.0
        elif result == -1:
            not_face = not_face + 1.0

print "Error on Faces: %.2f" % (not_face/(face + not_face))

face = 0.0;
not_face = 0.0;

for f in os.listdir('Test/Not_Face/'):
    if f.endswith(".JPEG"):
        filename = join(abspath('Test/Not_Face/'),f)
        image = cv2.imread(filename,0)
        blur = cv2.blur(image,(5,5))
        laplacian = cv2.Laplacian(blur,cv2.CV_64F)
        edge = cv2.Canny(blur,100,200)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
        hogF = hog(blur)
        #imhog = hog.compute(image)
        img = np.array(laplacian, dtype = np.float32).flatten()

        result = svm.predict(img)
        
        if result==1:
            face = face + 1.0
        elif result==-1:
            not_face = not_face + 1.0

print "Error on Not Faces: %.2f" % (face/(face + not_face))