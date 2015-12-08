__author__ = 'harshad'

import numpy as np
import scipy
from scipy import sparse
import math
import sys
import matplotlib.pyplot as plt

def calculateGain(info_a, info_b):
    gain = float(info_a) - float(info_b)
    return gain

def calculateEntropy(a,b):
    if a + b == 0 or float(a) + float(b) == 0.0 or b == 0:
        return 0

    if a == 0 or float(a) == 0.0:
        p_a = 0.0
        log_a = 0.0
    else:
        p_a = float(a)/float(a+b)
        log_a = math.log(p_a,2)

    if b == 0 or float(b) == 0.0:
        p_b = 0.0
        log_b = 0.0
    else:
        p_b = float(b)/float(a+b)
        log_b = math.log(p_b,2)

    entropy = -(p_a*log_a) - (p_b*log_b)
    return entropy

def loadData(path):
    matrix = np.loadtxt(path)
    extracted_truths = matrix[:,len(matrix[0])-1]
    ''' Deal with the truth(s) [#pun intended], maintain (M x 1) shape.'''
    print 'inside loadData, truths shape= ',extracted_truths.shape
    truths = np.zeros(shape=(len(extracted_truths),1))
    count = 0
    for i in range(len(extracted_truths)):
        truths[i,0] = extracted_truths[i]

    matrix = np.delete(matrix,len(matrix[0])-1,axis=1)
    print 'refactored shape of truths = ',truths.shape
    return matrix, truths

def discretizeData(matrix):
    for each_column in matrix.T:
        mean = np.mean(each_column)
        std = np.std(each_column)
        list,edges = np.histogram(each_column,bins=5)
        # plt.hist(each_column,bins=5)

        for i in range(len(each_column)):
            if edges[0] <= each_column[i] < edges[1]:
                each_column[i] = 1
            elif edges[1] <= each_column[i] < edges[2]:
                each_column[i] = 2
            elif edges[2]<= each_column[i] < edges[3]:
                each_column[i] = 3
            elif edges[3] <= each_column[i] < edges[4]:
                each_column[i] = 4
            elif edges[4]<= each_column[i] <= edges[5]:
                each_column[i] = 5

    print 'len of matrix = ',len(matrix),len(matrix[0])
    print 'len of truths = ',len(truths)
    return matrix

def calculateColGains(matrix,truths):
    '''
    calculateColGains() takes the matrix and the truths and calculates gains for each column
    (that is, each feature) and returns a list of gains. each index represents a feature
    for which the gain is calculated.
    '''
    # truths = matrix[:,len(matrix[0])-1]
    matrix = np.delete(matrix,len(matrix[0])-1,axis=1)
    total_entropy = calculateEntropy(list(truths).count(0), list(truths).count(1))
    gains_list = [ ]

    j = 0
    for each_column in matrix.T:
        feature_map = generateFeatureMap(each_column,truths)
        gain = 0.0

        for key in feature_map:
            values = feature_map[key]
            count0 = values[0]
            count1 = values[1]

            feature_entropy = calculateEntropy(count0,count1)

            priori_denom = len(truths)
            priori_num = count0 + count1
            priori = float(priori_num)/float(priori_denom)

            gain = gain + float(priori) * float(feature_entropy)

        gains_list.insert(j,(float(total_entropy) - float(gain)))
        j = j + 1

    return gains_list

def generateFeatureMap(feature_list,labels):
    '''
    generateFeatureCount() generates a map which has
    each feature and its label0_count and label1_count.
    '''
    print 'feature lsit = ', feature_list
    print 'label list = ', labels.shape

    feature_map = { }
    feature_set = np.unique(feature_list)

    print 'feature set = ', feature_set

    for i in range(5):
        feature_map[i] = (0,0)

    j = 0
    for each in feature_set:
        count0 = 0
        count1 = 0
        for k in range(len(feature_list)):
            if feature_list[k] == each and labels[k,0] == 0:
                count0 = count0 + 1
            elif feature_list[k] == each and labels[k,0] == 1:
                count1 = count1 + 1
        # print 'for feature = ',each,'counts = ',count0,',',count1
        feature_map[each] = (count0,count1)

    return feature_map

if __name__ == '__main__':
    '''toy matrix part for experimenting'''
    # matrix = np.array([[1,1,1],[2,1,2],[3,2,2,]])
    # truths = np.array([[0],[1],[1]])
    # hmap = generateFeatureMap(matrix[:,0],truths)
    '''toy matrix part ends'''

    matrix,truths = loadData('data/dataset1.txt') # Real Deal

    # print matrix,truths
    matrix = discretizeData(matrix)
    print truths.shape
    matrix = np.column_stack((matrix,truths))

    '''calculate total entropy'''
    total_entropy = calculateEntropy(list(truths).count(0),list(truths).count(1))
    print total_entropy

    '''pass the matrix to calculate all gains, to determine which feature is the root'''
    gains_list = calculateColGains(matrix,truths)

    print gains_list.index(max(gains_list))
    print 'second root = ', gains_list.index(sorted(gains_list)[1])




