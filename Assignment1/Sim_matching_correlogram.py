# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:25:06 2020

@author: Pranjal
"""

import os
import numpy as np
import cv2
import pickle
import operator
import time



def import_Images(image_dataset):
    images_List = os.listdir("/content/drive/My Drive/HW-1/Correlogram_Features2/")
    index = 0
    print(len(images_List))
    for feature in images_List:

        # if(index > 20):
        #   return
          file = []
          with open("/content/drive/My Drive/HW-1/Correlogram_Features2/" + feature,"rb") as f:
              file = pickle.load(f)
          #print(np.array(file).shape)
          # if(np.shape(np.array(file)) == (5, 256, 1)):
          #   index+=1
          #   image_dataset[feature] = np.reshape(np.array(file), (5, -1))
          if(np.shape(np.array(file)) != (5, 512, 1)):
            print(np.shape(np.array(file)))
          image_dataset[feature] = np.reshape(np.array(file), (5, -1))
          index+=1
          if(index%300==0):
            print(index)

def distance_metric(cur_subj, query, metric):
    if(metric == 0):
      dist = 0
      diff_matrix = np.array(cur_subj - query)
      for i in range(diff_matrix.shape[0]):
          for j in range(diff_matrix.shape[1]):
              dist+=pow(diff_matrix[i][j], 2)
      return  dist
    elif(metric == 1):
      return np.sum(np.absolute(cur_subj-query))

   
def find_Distances(query, images, metric):
    distance_from_Query = {}
    for image in images:
        distance_from_Query[image] =  distance_metric(images[image], query, metric)
    return distance_from_Query

def get_top(num_images, distance_from_Query):
  index = 0
  top_list = [] 
  for image in distance_from_Query:
    if(index == num_images):
      break
    else:
      top_list.append(image)
    index+=1
  return top_list

def match_good_ok(top_list, query):
    correct_good = []
    correct_ok = []
    correct_junk = []
    num_correct_good = 0
    num_correct_ok = 0
    num_correct_junk = 0

    num_correct = 0
    with open("/content/drive/My Drive/HW-1/train/ground_truth/" + query + "_ok.txt","r") as f:
      for x in f:
        x = x[0:x.find("\n")]
        correct_ok.append(x)
    with open("/content/drive/My Drive/HW-1/train/ground_truth/" + query + "_good.txt","r") as f:
      for x in f:
        x = x[0:x.find("\n")]
        correct_good.append(x)
    with open("/content/drive/My Drive/HW-1/train/ground_truth/" + query + "_junk.txt","r") as f:
      for x in f:
        x = x[0:x.find("\n")]
        correct_junk.append(x)
    # print("Length of correct ", len(correct))
    for image in top_list:
      # print("image from top list", image[0])
      for image2 in correct_good:
        # print("image from correct", image2+".txt")
        if(image[0] == (image2+".txt")):
          num_correct_good+=1
          num_correct+=1
          break

    for image in top_list:
      # print("image from top list", image[0])
      for image2 in correct_ok:
        # print("image from correct", image2+".txt")
        if(image[0] == (image2+".txt")):
          num_correct_ok+=1
          num_correct+=1
          break

    for image in top_list:
      # print("image from top list", image[0])
      for image2 in correct_junk:
        # print("image from correct", image2+".txt")
        if(image[0] == (image2+".txt")):
          num_correct_junk+=1
          num_correct+=1
          break
    return num_correct, len(correct_good) + len(correct_ok) + len(correct_junk), num_correct_good/float(len(correct_good)), num_correct_ok/float(len(correct_ok)), num_correct_junk/float(len(correct_junk))  
     
image_dataset = {}  
import_Images(image_dataset)
print("Imported", len(image_dataset.keys()))


query_list = os.listdir("/content/drive/My Drive/HW-1/train/query/")
max_prec = [0, 0]
max_recall = [0, 0]
sum_prec =[0, 0]
sum_recall = [0, 0]
f1_score = [0, 0]
max_perc_good = [0, 0]
max_perc_ok = [0, 0]
max_perc_junk = [0, 0]
sum_time = [0, 0]
num_top = 100
for query_file in query_list:
  query = ""
  query2 = ""
  with open("/content/drive/My Drive/HW-1/train/query/" + query_file,"rb") as f:
    # print(f.read().split()[0][5:].decode('utf-8'))
    query = f.read().split()[0][5:].decode('utf-8') + ".txt"
    query2 = query_file.split("_query")[0]

# query = "christ_church_000179.txt"


query_list = os.listdir("/content/drive/My Drive/HW-1/train/query/")
max_prec = [0, 0]
max_recall = [0, 0]
min_prec = [10, 10]
min_recall = [10, 10]
sum_prec =[0, 0]
sum_recall = [0, 0]
max_f1_score = [0, 0]
min_f1_score = [10, 10]
max_perc_good = [0, 0]
max_perc_ok = [0, 0]
max_perc_junk = [0, 0]
sum_time = [0, 0]
num_top = 50
image_highest_f1 = ""
image_lowest_f1 = ""
prec_arr = [[], []]
recall_arr = [[], []]
f1_arr = [[], []]

for query_file in query_list:
  query = ""
  query2 = ""
  with open("/content/drive/My Drive/HW-1/train/query/" + query_file,"rb") as f:
    # print(f.read().split()[0][5:].decode('utf-8'))
    query = f.read().split()[0][5:].decode('utf-8') + ".txt"
    query2 = query_file.split("_query")[0]
    # print(query2)

# query = "christ_church_000179.txt"

  try:
    for i in range(0, 2):
      distance_from_Query = {}
      time_initial = time.clock()
      distance_from_Query = find_Distances(image_dataset[query], image_dataset, i)
      sorted_distance = sorted(distance_from_Query.items(), key=operator.itemgetter(1))
      top_list = get_top(num_top, sorted_distance)
      # query2 = "christ_church_1"
      num_matches, num_given, perc_correct_good, perc_correct_ok, perc_correct_junk = match_good_ok(top_list, query2)
      time_end = time.clock()
      sum_time[i]+=(time_end - time_initial)
      max_perc_good[i] = max(max_perc_good[i], perc_correct_good)
      max_perc_ok[i] = max(max_perc_ok[i], perc_correct_ok)
      max_perc_junk[i] = max(max_perc_junk[i], perc_correct_junk)
      prec = float(num_matches)/num_top
      recall = float(num_matches)/num_given
      f1 = 2*prec*recall/float(prec + recall)
      if(f1 > max_f1_score[i]):
        image_highest_f1 = query
      if(f1 < min_f1_score[i]):
        image_lowest_f1 = query
      # sum_prec[i]+=prec
      # sum_recall[i]+=recall
      max_f1_score[i] = max(max_f1_score[i], f1)
      min_f1_score[i] = min(min_f1_score[i], f1)
      prec_arr[i].append(prec)
      recall_arr[i].append(recall)
      f1_arr[i].append(f1)
      # max_prec[i] = max(max_prec[i], prec)
      # max_recall[i] = max(max_recall[i], recall)
      # min_prec[i] = min(min_prec[i], prec)
      # min_recall[i] = min(min_recall[i], recall)
      # print(query,"Metric" + str(metric),  num_matches)
  except Exception as e: 
    print(e)
    continue

sum_time[0]/=len(query_list)
sum_time[1]/=len(query_list)

prec_arr[0].sort(reverse = True)
prec_arr[1].sort(reverse = True)
recall_arr[0].sort(reverse = True)
recall_arr[1].sort(reverse = True)
f1_arr[0].sort(reverse = True)
f1_arr[1].sort(reverse = True)
print("max_prec_1",prec_arr[0][0])
print("max_prec_2",prec_arr[1][0])
print("max_recall_1",recall_arr[0][0])
print("max_recall_2",recall_arr[1][0])
print("max_f1_1",f1_arr[0][0])
print("max_f1_2",f1_arr[1][0])
print("avg_recall_1", sum(recall_arr[0])/float(len(query_list)))
print("avg_recall_1", sum(recall_arr[1])/float(len(query_list)))
print("avg_prec_1", sum(prec_arr[0])/float(len(query_list)))
print("avg_prec_1", sum(prec_arr[1])/float(len(query_list)))
print("avg_f1_1", sum(f1_arr[0])/float(len(query_list)))
print("avg_f1_1", sum(f1_arr[1])/float(len(query_list)))
print(image_highest_f1)
print(image_lowest_f1)
print("perc_good", max_perc_good)
print("perc_ok", max_perc_ok)
print("perc_junk", max_perc_junk)
print("average_time", sum_time)
print("min prec", prec_arr[0][len(prec_arr[0]) - 1])
print("min prec", prec_arr[1][len(prec_arr[1]) - 1])
print("min recall", recall_arr[0][len(recall_arr[0]) - 1])
print("min recall", recall_arr[1][len(recall_arr[1]) - 1])
print("min f1", f1_arr[0][len(f1_arr[0]) - 1])
print("min f1", f1_arr[1][len(f1_arr[1]) - 1])







