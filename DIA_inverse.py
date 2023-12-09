import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from PIL import Image

# Load data from LRDS.mat
data = loadmat('LRDS.mat')
LRDS = data['LRDS']

# Read and preprocess test image
I = np.array(Image.open('test_img_6.bmp').resize((512, 512)))
I = I.astype(float) / np.max(I)

#O0 is propto E
O0 = np.sqrt(I)

# ASM (angular spectrum method) parameters
w = 0.650 * 10**(-3)  # wavelength
dx = 7.56 * 10**(-3)  # DMD pixel pitch
f = 200
M = 2048
num_iter = 1
X, Y = np.meshgrid(np.arange(1, M+1), np.arange(1, M+1))
deltaf = 1 / (M * dx)
SFTF = np.exp(-2j * np.pi * f * ((1/w)**2 - ((Y-M/2-1) * deltaf)**2 - ((X-M/2-1) * deltaf)**2)**0.5)
A = np.zeros((M, M))
Hbr = np.zeros((M, M))
error = 0.8
mask2 = np.ones((512, 512))
MSE = np.zeros(num_iter)
PSNR = np.zeros(num_iter)
Itotal = 0
A = np.zeros((M, M))
Hbr = np.zeros((M, M))

# Loop over LRDS
for i in range(num_iter):
    randmask = LRDS[:, :, i]
    O = randmask * O0
    # randomly mask the imput intensity O0
    A[1601-256:1601+256, 1601-256:1601+256] = O
    H = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(A))) * SFTF))
    #make the mask binary
    Hb = np.real(H) > 0
    Hb = Hb[1025-256:1024+256, 1025-256:1024+256]
    Hbr[1025-256:1024+256, 1025-256:1024+256] = Hb
    #generate the image using FFT
    E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Hbr))) * np.conj(SFTF)))
    II = np.abs(E)**2
    Itotal = Itotal + II
    I1 = (np.mean(np.mean(I))) / (np.mean(np.mean(Itotal[1601-256:1601+256, 1601-256:1601+256]))) * Itotal[1601-256:1601+256, 1601-256:1601+256]
    MSE[i] = (np.sum(np.sum((I - I1)**2))) / (512 * 512)

    #calculate signal to noise ratio
    PSNR[i] = 10 * np.log10((np.max(I)) / MSE[i])
    print(MSE[i])
    print(np.sqrt(MSE[i])*100)
    print(PSNR[i])

# Save Hb.mat
savemat('Hb.mat', {'Hb': Hb})

# Plot PSNR
# x = np.linspace(1, num_iter, num_iter)
# plt.plot(x, PSNR)
# plt.xlabel('Number of holograms')
# plt.ylabel('PSNR(dB)')
# plt.show()

# Display Hb
# cc = Hb[:, :]
# plt.imshow(cc, cmap = "gray")
# plt.show()

