import librosa
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score




dict = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9} 

file = "/content/drive/My Drive/HW2/Dataset/Spectogram"
spec_dir_list = os.listdir(file)
print(len(spec_dir_list))
X = []
Y = []
for dir in spec_dir_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  spec_list = os.listdir(cur_dir)
  for spec_file in spec_list:
    spec = np.loadtxt(cur_dir + "/" + spec_file)
    # print(spec.shape)
    spec_flattened = np.zeros((161*99))
    spec = spec.flatten()
    spec_flattened[:len(spec)] = spec
    X.append(spec_flattened)
    Y.append(dict[dir])

print("Train import done")

file = "/content/drive/My Drive/HW2/Dataset/Spectogram_Val"
spec_dir_list = os.listdir(file)
print(len(spec_dir_list))
X_test = []
Y_test = []
for dir in spec_dir_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  spec_list = os.listdir(cur_dir)
  for spec_file in spec_list:
    spec = np.loadtxt(cur_dir + "/" + spec_file)

    spec_flattened = np.zeros((161*99))
    spec = spec.flatten()
    spec_flattened[:len(spec)] = spec
    X_test.append(spec_flattened)
    Y_test.append(dict[dir])

print("Test import done")

from sklearn.utils import shuffle
X, Y = shuffle(X, Y)

X_test, Y_test = shuffle(X_test, Y_test)

print("X", len(X))
print("Y", len(Y))
print("X_test", len(X_test))
print("Y_test", len(Y_test))

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, Y)

Y_pred_svm_1 = classifier.predict(X_test)
print("SVM: ")
prec = precision_score(classifier.predict(X_test),Y_test, average='weighted')
rec = recall_score(classifier.predict(X_test),Y_test, average='weighted')
print("Test Prec ", prec, "Recall", rec, "f1 ",2*prec*rec/float(prec+rec))
# print("Test Noise: " + str(classifier.score(X_test_Noise,Y_test_Noise)))

# print("Spectogram done ---------------------------------------------------------")


file = "/content/drive/My Drive/HW2/Dataset/mfcc"
spec_dir_list = os.listdir(file)
print(len(spec_dir_list))
X = []
Y = []
for dir in spec_dir_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  spec_list = os.listdir(cur_dir)
  for spec_file in spec_list:
    spec = np.loadtxt(cur_dir + "/" + spec_file)
    spec_flattened = np.zeros((99*13))
    spec = spec.flatten()
    spec_flattened[:len(spec)] = spec
    # print(spec_flattened)
    X.append(spec_flattened)
    Y.append(dict[dir])

print("Train import done")

file = "/content/drive/My Drive/HW2/Dataset/mfcc_val"
spec_dir_list = os.listdir(file)
print(len(spec_dir_list))
X_test = []
Y_test = []
for dir in spec_dir_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  spec_list = os.listdir(cur_dir)
  for spec_file in spec_list:
    spec = np.loadtxt(cur_dir + "/" + spec_file)
    spec_flattened = np.zeros((99*13))
    spec = spec.flatten()
    spec_flattened[:len(spec)] = spec

    X_test.append(spec_flattened)
    Y_test.append(dict[dir])


from sklearn.utils import shuffle
X, Y = shuffle(X, Y)
X_test, Y_test = shuffle(X_test, Y_test)
# X_test_Noise, Y_test_Noise = shuffle(X_test_Noise, Y_test_Noise)

print("X", len(X))
print("Y", len(Y))
print("X_test", len(X_test))
print("Y_test", len(Y_test))
# print("X_test_Noise", len(X_test_Noise))
# print("Y_test_Noise", len(Y_test_Noise))

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, Y)

Y_pred_svm_1 = classifier.predict(X_test)
print("SVM: ")
print(classification_report(Y_test,Y_pred_svm_1))
prec = precision_score(classifier.predict(X_test),Y_test, average='weighted')
rec = recall_score(classifier.predict(X_test),Y_test, average='weighted')
print("Test Prec ", prec, "Recall", rec, "f1 ",2*prec*rec/float(prec+rec))
# print("Test Noise: " + str(classifier.score(X_test_Noise,Y_test_Noise)))





