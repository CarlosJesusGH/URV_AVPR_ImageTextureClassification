# cd ~/Dropbox/Personal/Study/MasterDegreeArtificialIntelligence/1st Semester/AVPR/AVPR - 2nd Project
# source ~/Coding/venv/bin/activate

# Packages for math and plot -----------------------
# pip install numpy
# pip install matplotlib
# Package for image processing ---------------------
# pip install -U scikit-image
# Packages for clustering --------------------------
# pip install scipy ipython jupyter pandas sympy nose

import matplotlib.pyplot as plt
import numpy as np
# from skimage.feature import greycomatrix, greycoprops
# from skimage import data, io
import skimage.io as ski_io
import skimage.feature as ski_feature
from skimage import color
import os
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import cv2



# Directory for reading
inputDir = "Images Exer2/grup1/"
listFiles = os.listdir(inputDir)
listFiles.sort()

glcmStep = 6
print "Using step= %d" % glcmStep

for i in range(1,10):   #5
    features = []
    for fileName in listFiles:
        # fileName = 'Burlap-1.png'
        # print "file: " + fileName
        I = ski_io.imread(inputDir + fileName);
        Icv = cv2.imread(inputDir + fileName);

        # print cv2.mean(Icv)[0]
        meanStdDev = cv2.meanStdDev(Icv)
        mean, stds = meanStdDev[0][0][0], meanStdDev[1][0][0]
        # print mean, stds
        # print I
        # GLCM = ski_feature.greycomatrix(I, [1], [0]);
        # print "max= %d" % I.max()
        # Good results using step=3
        # GLCM = ski_feature.greycomatrix(I, [i], [np.pi/2], levels=256, symmetric=True, normed=True)
        GLCM = ski_feature.greycomatrix(I, [glcmStep], [0], levels=256, symmetric=True, normed=True)

        # Get fft
        f = np.fft.fft2(I)
        fshift = np.fft.fftshift(f)
        imgAbs = np.abs(fshift)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        # print magnitude_spectrum

        # img_gray = cv2.cvtColor(imgAbs, cv2.COLOR_BGR2GRAY)
        # img_gray = np.dot(imgAbs[...,:3], [0.299, 0.587, 0.114])
        img_gray = color.rgb2gray(imgAbs)

        # plt.imshow(img_gray)
        # plt.show()
        #
        meanStdDevFFT = cv2.meanStdDev(img_gray)
        meanFFT, stdsFFT = meanStdDevFFT[0][0][0], meanStdDevFFT[1][0][0]
        # print meanFFT, stdsFFT

        # Get properties from GLCM
        props = {}
        # props['step'] = i
        # props['mean'] = mean
        # props['stds'] = stds
        props['meanFFT'] = meanFFT
        # props['stdsFFT'] = stdsFFT
        props['contrast'] = ski_feature.greycoprops(GLCM,'contrast')[0][0]  # Good
        # props['dissimilarity'] = ski_feature.greycoprops(GLCM,'dissimilarity')[0 ][0]     # Bad
        # props['homogeneity'] = ski_feature.greycoprops(GLCM, 'homogeneity')[0][0]         # Bad
        # props['energy'] = ski_feature.greycoprops(GLCM, 'energy')[0][0]   # Maybe
        # props['correlation'] = ski_feature.greycoprops(GLCM, 'correlation')[0][0]       # Good
        # props['ASM'] = ski_feature.greycoprops(GLCM, 'ASM')[0][0]           # Good

        features.append(props.values())

    features = np.asarray(features)

    # Print stack of features
    np.set_printoptions(precision=4, suppress=True)

    # print props.keys(), "\n", features

    # Apply clustering k-means
    # whitened = whiten(features)
    features_norm = features / features.max(axis=0)
    # print features_norm
    # book = np.array((whitened[0],whitened[2]))
    # print book
    kmeansRes = kmeans2(features_norm,6, minit='points', iter=100000)
    # print kmeansRes[0]  # Centroids
    print kmeansRes[1]

# Print script end
print("script ended successfully")
