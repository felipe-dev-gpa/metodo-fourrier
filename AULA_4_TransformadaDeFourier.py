import numpy as np
import cv2
from matplotlib import pyplot as plt

# Carregar as imagens
img1 = cv2.imread('imagem3.jpeg', 0)
img2 = cv2.imread('imagem2.jpeg', 0)
img3 = cv2.imread('imagem1.jpeg', 0)

# Verificar se as imagens foram carregadas corretamente
if img1 is None or img2 is None or img3 is None:
    print("Erro ao carregar uma ou mais imagens.")
else:
    sizes = 64
    img1_s = img1
    img2_s = img2[0:sizes, 0:sizes]
    img3_s = img3[5:sizes+5, 5:sizes+5]

    F1s = cv2.dft(np.float32(img1_s), flags=cv2.DFT_COMPLEX_OUTPUT)
    F2s = cv2.dft(np.float32(img2_s), flags=cv2.DFT_COMPLEX_OUTPUT)
    F3s = cv2.dft(np.float32(img3_s), flags=cv2.DFT_COMPLEX_OUTPUT)

    n2 = F1s.shape[0] // 2
    m2 = F1s.shape[1] // 2

    dft_shift1 = np.fft.fftshift(F1s)
    dft_shift2 = np.fft.fftshift(F2s)
    dft_shift3 = np.fft.fftshift(F3s)

    magnitude_spectrum1 = 20 * np.log(cv2.magnitude(dft_shift1[:, :, 0], dft_shift1[:, :, 1]))
    magnitude_spectrum2 = 20 * np.log(cv2.magnitude(dft_shift2[:, :, 0], dft_shift2[:, :, 1]))
    magnitude_spectrum3 = 20 * np.log(cv2.magnitude(dft_shift3[:, :, 0], dft_shift3[:, :, 1]))

    plt.figure(figsize=(12, 12))
    plt.subplot(331)
    plt.imshow(img1_s, cmap="gray")
    plt.axis('off')
    plt.title('Original 1')
    plt.subplot(332)
    plt.imshow(img2_s, cmap="gray")
    plt.axis('off')
    plt.title('Original 2')
    plt.subplot(333)
    plt.imshow(img3_s, cmap="gray")
    plt.axis('off')
    plt.title('Original 3')
    plt.subplot(334)
    plt.imshow(magnitude_spectrum1, cmap="gray")
    plt.axis('off')
    plt.title('Filtered 1')
    plt.subplot(335)
    plt.imshow(magnitude_spectrum2, cmap="gray")
    plt.axis('off')
    plt.title('Filtered 2')
    plt.subplot(336)
    plt.imshow(magnitude_spectrum3, cmap="gray")
    plt.axis('off')
    plt.title('Filtered 3')

    rows1 = F1s.shape[0] // 2
    cols1 = F1s.shape[1] // 2

    F1p = np.fft.fftshift(F1s).copy()
    F1p[:n2-9, :] = 0  # square low pass filter, removes higher frequencies
    F1p[:, :m2-9] = 0  # square low pass filter, removes higher frequencies
    F1p[n2+9:, :] = 0  # square low pass filter, removes higher frequencies
    F1p[:, m2+9:] = 0  # square low pass filter, removes higher frequencies
    F1p = np.fft.ifftshift(F1p)

    img_back = cv2.idft(F1p)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(337)
    plt.imshow(img1_s, cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(338)
    plt.imshow(img_back, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
