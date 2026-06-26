import pyvista as pv
import numpy as np
from skimage import measure
import h5py
import cv2
from tqdm import tqdm

data1 = h5py.File(r'D:\pycode\BP-AdMT\data\LA\0RZDK210BSMWAA6467LU\mri_norm2.h5', 'r') # 48
data2 = h5py.File(r'D:\pycode\BP-AdMT\data\LA\1D7CUD1955YZPGK8XHJX\mri_norm2.h5', 'r') # 62
image, label = np.array(data2['image']), np.array(data2['label'])
label_list = []
# for i in tqdm(range(image.shape[2])):
#         if np.sum(label[:,:,i]) > 0:
#             label_list.append(i)
# print(label_list)
# print(np.unique(image))
# print(np.unique(label))

image = (image-np.min(image)) / (np.max(image)-np.min(image))
image = (image*255).astype(np.uint8)
label = (label*255).astype(np.uint8)
print(image.shape, label.shape)
cv2.imwrite(rf'LA2_image.png',image[:,:,62])
cv2.imwrite(rf'LA2_label.png',label[:,:,62])
# BraTS
# for i in tqdm(range(image.shape[0])):
#         contact = np.hstack([image[i,:,:], label[i,:,:]])
#         cv2.imwrite(rf'./data/BraTS/contact{i}.png', contact)
#         # cv2.imwrite(rf'./BraTS/data/contact{i}.png', image[:,:,i])
# LA
# for i in tqdm(range(image.shape[2])):
#         contact = np.hstack([image[:,:,i], label[:,:,i]])
#         cv2.imwrite(rf'./data/LA2/contact{i}.png', contact)
#         # cv2.imwrite(rf'./BraTS/data/contact{i}.png', image[:,:,i])
# Pancreas
# for i in tqdm(range(image.shape[2])):
#         contact = np.hstack([image[:,:,i], label[:,:,i]])
#         cv2.imwrite(rf'./data/Pancreas/contact{i}.png', contact)
        # cv2.imwrite(rf'./BraTS/data/contact{i}.png', image[:,:,i])
# i = 50
# image = image[:,:,i]
# label = label[:,:,i]
# contact = np.hstack([image, label])
# cv2.imshow('image', image)
# cv2.imshow('label', label)
# cv2.imshow('contact', contact)
# cv2.waitKey(0)
# cv2.destroyAllWindows()