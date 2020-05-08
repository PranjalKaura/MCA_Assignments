import os
import numpy as np
import cv2
from scipy import ndimage

def Normalise(image):
    return image/255

def Blob_Detector(image, sigma):
    filtered_Image = ndimage.filters.gaussian_laplace(image, sigma)
    return filtered_Image
  
def Non_maxima_Supr(feature_Space1, threshold, ActualImage):
    image = np.zeros((feature_Space1.shape))
    for i in range(0, image.shape[0]):
        image[i] = ndimage.filters.rank_filter(feature_Space1[i], -1, size = (5, 5), mode = 'constant')
        
    SupressedImage = [] 
    for x in range(feature_Space1.shape[1]):
        for y in range(feature_Space1.shape[2]):
            pixel_Val = -1000
            for scale in range(image.shape[0]):
                pixel_Val = max(pixel_Val, image[scale][x][y])
            if(pixel_Val > threshold):
                SupressedImage.append([pixel_Val, ActualImage[x][y], x, y])   
    return extract_Top_features(np.array(SupressedImage), 2000)

def extract_Top_features(pixel_arr, num_features):
  if(len(pixel_arr) == 0):
    return []
  if(len(pixel_arr) < num_features):
    return pixel_arr[:, [1]]
  pixel_arr = np.array(sorted(pixel_arr, key=lambda x: x[0], reverse=True))
  return pixel_arr[0:num_features, [1, 2, 3]]

def Blob_changing_Sigma(image, sigma, k, size_scale):
    feature_Space = np.empty((size_scale, image.shape[0], image.shape[1]))
    for i in range(0, size_scale):
        sigma_updated = (pow(k, i))*sigma
        filtered_Image = Blob_Detector(image, sigma_updated)
        feature_Space[i] = filtered_Image
    
    return feature_Space

images_List = os.listdir('/content/drive/My Drive/HW-1/images')
print("Num Images", len(images_List))

for i in range(0, len(images_List)):
  fileName = images_List[i]
  image = Normalise(np.array(cv2.imread("/content/drive/My Drive/HW-1/images/" + fileName, 0)))
  image = cv2.resize(image, (512, 512))
  k = 1.5
  sigma = 1
  fileName = fileName.split(".")[0]
  feature_Space1 = Blob_changing_Sigma(image, sigma, k, 10)
  feature_Space2 = Non_maxima_Supr(feature_Space1, 0.01, image)
  # print(np.array(feature_Space2).shape)
  # print(feature_Space2[1])
  np.savetxt("/content/drive/My Drive/HW-1/Blob_Features3/" + fileName + ".txt", feature_Space2)
  if(i%300 == 0):
    print(i, len(feature_Space2))

#ref https://projectsflix.com/opencv/laplacian-blob-detector-using-python/
