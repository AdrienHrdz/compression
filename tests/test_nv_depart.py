# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import sys, os
import gpt3_functions.gpt3_functions as gpt3_functions

# %%
# Chargement de l'image
filename = './figures/06.png'
IMAGE_bgr = cv2.imread(filename)
IMAGE_rgb = cv2.cvtColor(IMAGE_bgr, cv2.COLOR_BGR2RGB)
IMAGE_nb = cv2.cvtColor(IMAGE_rgb, cv2.COLOR_RGB2GRAY)
[H,W] = IMAGE_nb.shape

plt.subplot(121)
plt.imshow(IMAGE_rgb)
plt.subplot(122)
plt.imshow(IMAGE_nb, cmap='gray')
cv2.imwrite('06_nb.png', IMAGE_nb)


plt.show()

# %%
mat_corr = np.corrcoef(IMAGE_nb)
plt.imshow(mat_corr)
plt.colorbar()
plt.show()

# %%
# separate IMAGE_nb into 8x8 blocks
def decoupage88_copilot(mat):
    listImages88 = list()
    for i in range(0, W, 8):
        for j in range(0, H, 8):
            imagette = mat[i:i+8, j:j+8]
            listImages88.append(imagette)
    return listImages88
# reconstruct IMAGE_nb from 8x8 blocks
def reconstruct88_copilot(pList):
    pList_tmp = np.copy(pList).tolist()
    mat = np.zeros((H,W))
    for i in range(0, W, 8):
        for j in range(0, H, 8):
            mat[i:i+8, j:j+8] = pList_tmp.pop(0)
    return mat

# %%
listBlocks88 = decoupage88_copilot(IMAGE_nb)
Irec = reconstruct88_copilot(listBlocks88)
plt.imshow(Irec, cmap='gray')
plt.show()

# %%
# Calcul de la DCT par blocs de 8x8
listBlocks88_dct = list()
for bloc in listBlocks88:
    dct_block = cv2.dct(bloc.astype(np.float32))
    listBlocks88_dct.append(dct_block)
dct_par_block = reconstruct88_copilot(listBlocks88_dct)
plt.imshow(10*(np.abs(dct_par_block)), cmap='gray')
plt.show()

# %%
Z = [[16, 11, 10, 16, 24,  40,  51,  61],
     [12, 12, 14, 19, 26,  58,  60,  55],
     [14, 13, 16, 24, 40,  57,  69,  56],
     [14, 17, 22, 29, 51,  87,  80,  62],
     [18, 22, 37, 56, 68,  109, 103, 77],
     [24, 35, 55, 64, 81,  104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]]
Z = np.ones((8,8))
Z = np.array(Z)

T_hat = list()
T_hat_posNonNul = list()

for T in listBlocks88_dct:
    pass

# # %%
# # Decodage de l'image
# listBlocks88_quantifie = list()
# for T in T_hat:
#     listBlocks88_quantifie.append(T*Z) 

# # %%
# # Reconstruction de l'image
# Irec = reconstruct88_copilot(listBlocks88_quantifie)
# plt.imshow(Irec, cmap='gray')
# plt.show()