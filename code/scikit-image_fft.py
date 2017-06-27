
import numpy as np
from numpy import fft
from skimage import io, data
import matplotlib.pyplot as plot


# im = data.coffee()
# im = io.imread('Images Exer2/grup1/Burlap-1.png');
# f = fft.fft2(im)
# f2 = fft.ifft2(f)
# r = numpy.real(f2)


img = io.imread('Images Exer2/grup1/Burlap-1.png');

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plot.imshow(magnitude_spectrum, cmap = 'gray')
plot.show()
