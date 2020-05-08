# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:18:41 2020

@author: Pranjal
"""

import numpy as np
import cv2
import os
import pickle
            

def Map_to_bin(image, num_bins):
    val = num_bins/256.0
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            image[x][y][0] = int((image[x][y][0]*val))
            image[x][y][1] = int((image[x][y][1]*val))
            image[x][y][2] = int((image[x][y][2]*val))
    
    return image

def create_colors(num_bins):
    colors = np.zeros((int(pow(num_bins, 3)), 3))
    index = 0
    for r in range(0, num_bins):
        for g in range(0, num_bins):
            for b in range(0, num_bins):
                colors[index] = [r ,g, b]
                index+=1
    return colors


def correlogram(image):
    num_bins = 8
    image = Map_to_bin(image, num_bins)
    colors = create_colors(num_bins)

    distance = [3, 5, 7, 9, 12]

    numColors = len(colors)
    correlogram = []
    
    for i in range(0 , len(distance)):
        cur_dist = distance[i]
        color_Arr = np.zeros((colors.shape[0], 1))
        color_count = 0
        X = image.shape[0]
        for x in range(0, X, int(X/7)):
            Y = image.shape[1]
            for y in range(0, Y, int(Y/7)):
                points_pos = []
                cur_pixel = image[x][y]
                if(x - cur_dist>=0):
                    points_pos.append([x - cur_dist, y])
                    if(y - cur_dist >= 0):
                        points_pos.append([x - cur_dist, y - cur_dist])
                        points_pos.append([x, y - cur_dist])
                    if(y + cur_dist < Y):
                        points_pos.append([x - cur_dist, y + cur_dist])
                        points_pos.append([x, y + cur_dist])
                if(x + cur_dist < X):
                    points_pos.append([x + cur_dist, y])
                    if(y - cur_dist >= 0):
                        points_pos.append([x + cur_dist, y - cur_dist])
                    if(y + cur_dist < Y):
                        points_pos.append([x + cur_dist, y + cur_dist])
                 
                for z in range(0, len(points_pos)):
                    x_neigh = points_pos[z][0]
                    y_neigh = points_pos[z][1]
                    pixel_neigh = image[x_neigh][y_neigh]
                    for k in range(0, numColors):
                        if((pixel_neigh == cur_pixel).all() and (colors[k] == pixel_neigh).all()):
                            color_Arr[k]+=1
                            color_count+=1
                            break
        if(color_count == 0):
            continue
        for d in range(0, len(color_Arr)):
            color_Arr[d]/=color_count
        
        correlogram.append(color_Arr)
    return correlogram
        
images_List = os.listdir('/content/drive/My Drive/HW-1/images')
print("Num Images", len(images_List))
count = 0
for i in range(0, len(images_List)):
  count = i
  fileName = images_List[i]
  image = np.array(cv2.imread("/content/drive/My Drive/HW-1/images/" + fileName))
  image = cv2.resize(image, (512, 512))
  feature = correlogram(image)
  fileName = fileName.split(".")[0]
  with open("/content/drive/My Drive/HW-1/Correlogram_Features2/" + fileName + ".txt","wb") as f:
    pickle.dump(feature,f)
  """
  with open("myfile.pkl","rb") as f:
    feature = pickle.load(f)
  """  
  if(len(feature)!=5):
    print("ERROR", len(feature))

  if(i%300 == 0):
    print(i, len(feature))                      
                        
# discussed the solution with my mate priyanshiJain 2017358

    