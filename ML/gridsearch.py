import pickle
import numpy as np
from model import *
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    vector_size = [200, 300, 400]
    min_count = [3, 4, 5]
    window = [4, 6, 8]

    for x in vector_size:
        for y in min_count:
            for z in window:
                with open("/app/dataset/result" + "_" + str(x) + "_" + str(y) + "_" + str(z), 'rb') as f:
                    score = pickle.load(f)
                    print("score" + "_" + str(x) + "_" + str(y) + "_" + str(z) + " : ", score)

               

if __name__=="__main__":
    main()