__author__ = 'harshad'

''''
About script:
    Builds a decision tree and uses the test data to predict labels by traversing the tree.
'''

import numpy as np
import scipy
from scipy import sparse
import math
import sklearn
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
from DecisionTree import *
from Node import *
from copy import deepcopy

features_to_be_left = [ ]
global_list = [ ]

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
    original_matrix = matrix
    # indexes = np.random.permutation(matrix.shape[0])
    indexes = range(matrix.shape[0])
    partition = int(len(matrix) * 0.7)
    training_indexes, testing_indexes = indexes[:partition], indexes[partition:]
    print 'train indexes = ', training_indexes
    training, testing = matrix[training_indexes,:], matrix[testing_indexes,:]
    # print len(training), len(testing)
    training_truths_index = len(training[0]) - 1
    training_truths = training[:,training_truths_index]
    print 'length of train truths = ',len(training_truths)
    training_matrix = training
    training = np.delete(training,training_truths_index,1)
    print 'length of training columns = ', len(training[0])
    training_truths_to_return = np.zeros(shape=(len(training_truths),1))
    for i in range(len(training_truths)):
        training_truths_to_return[i,0] = training_truths[i]
    testing_truths_index = len(testing[0]) - 1
    testing_truths = testing[:,testing_truths_index]
    print 'length of test truths = ',len(testing_truths)
    testing = np.delete(testing,testing_truths_index,1)
    print 'length of test columns = ', len(testing[0])
    testing_truths_to_return = np.zeros(shape=(len(testing_truths),1))
    for i in range(len(testing_truths)):
        testing_truths_to_return[i,0] = testing_truths[i]
    return training,training_truths_to_return,testing,testing_truths_to_return

def discretizeData(matrix):
    for each_column in matrix.T:
        mean = np.mean(each_column)
        std = np.std(each_column)
        list,edges = np.histogram(each_column,bins=2)
        # plt.hist(each_column,bins=2)
        for i in range(len(each_column)):
            if edges[0] <= each_column[i] < edges[1]:
                each_column[i] = 0
            elif edges[1] <= each_column[i] <= edges[2]:
                each_column[i] = 1
    return matrix

def calculateColGains(matrix,truths):
    '''
    calculateColGains() takes the matrix and the truths and calculates gains for each column
    (that is, each feature) and returns a list of gains. each index represents a feature
    for which the gain is calculated. Must be called after discretizeData() method is called.
    '''
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
            denom = len(truths)
            num = count0 + count1
            try:
                weight = float(num)/float(denom)
            except ZeroDivisionError:
                print 'truths in cal',truths
                print 'matrix in cal', matrix.shape,matrix.size
            gain = gain + float(weight) * float(feature_entropy)
        gains_list.insert(j,(float(total_entropy) - float(gain)))
        j = j + 1
    return gains_list

def generateFeatureMap(feature_list,labels):
    '''
    generateFeatureCount() generates a map which has
    each feature and its label0_count and label1_count.
    '''
    feature_map = { }
    feature_set = np.unique(feature_list)
    for i in range(2):
        feature_map[i] = (0,0)
    j = 0
    for each in (0,1):
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

def inorder(root):
    if root == None:
        return
    print root.column_number
    inorder(root.left)
    inorder(root.right)

def calGain(matrix,truths):
    print 'in calgain'
    max_gains = [ ]
    max_gains_index = [ ]
    gains_values = [ ]
    global global_list
    j=0
    for each_column in matrix.T:
        gains_per_column = [ ]
        k = 0
        for feature in each_column:
            '''Calculate entropy for each feature
            number of vals less than x & number of vals greater than x
            '''
            less_than_x =  len([i for i in each_column if i <= feature])
            greater_than_x = len([i for i in each_column if i > feature])
            denom_total = len(each_column)
            less_weight = float(less_than_x)/float(denom_total)
            greater_weight = float(greater_than_x)/float(denom_total)
            less_than_0 = 0
            greater_than_0 = 0
            less_than_1 = 0
            greater_than_1 = 0
            p=0
            for val in each_column:
                if val <= feature and truths[p] == 0:
                    less_than_0 += 1
                elif val > feature and truths[p] == 0:
                    greater_than_0 += 1
                elif val <= feature and truths[p] == 1:
                    less_than_1 += 1
                elif val > feature and truths[p] == 1:
                    greater_than_1 += 1
                p += 1
            less_entropy = calculateEntropy(less_than_0,less_than_1)
            greater_entropy = calculateEntropy(greater_than_0,greater_than_1)
            to_subtract = (less_weight * less_entropy) + (greater_weight * greater_entropy)
            num_0 = list(np.ndarray.flatten(truths)).count(0)
            num_1 = list(np.ndarray.flatten(truths)).count(1)
            total_entropy = calculateEntropy(num_0,num_1)
            current_gain = total_entropy - to_subtract
            gains_per_column.insert(k,current_gain)
            k += 1
        '''find max gain for each column'''
        col_max_gain = max(gains_per_column)
        gains_values.insert(j,col_max_gain)
        index_of_col_max_gain = gains_per_column.index(col_max_gain)
        # print 'splitting value = ',each_column[gains_per_column.index(col_max_gain)],' & max gain val = ',col_max_gain
        max_gains.insert(j,each_column[gains_per_column.index(col_max_gain)])
        max_gains_index.insert(j,index_of_col_max_gain)
        j += 1
    global_list = max_gains
    print 'gains list = ', max_gains
    return max_gains,gains_values,max_gains_index

def dtpredictrecursion(each_row, root):
    if root.is_leaf == True:
        #print root.decision_value
        return root.decision_value
    else:
        if each_row[root.column_number] <= root.val:
            return dtpredictrecursion(each_row, root.left)
        elif each_row[root.column_number] > root.val:
            return dtpredictrecursion(each_row, root.right)

def dtpredict(root, testing, testing_truths):
    i=0;
    labels = np.zeros((testing.shape[0]))
    for each_row in testing:
        row_label = dtpredictrecursion(each_row, root)
        #print row_label
        labels[i] = row_label
        i += 1
    return labels

def loadDataset2(path):
    file = open(name=path,mode='r')
    content = file.read()
    lines = content.split('\n')
    # print type((lines[0].split())[3])
    cols = []
    for x in range(len(lines)):
        cols.append([])

    matrix = np.ndarray(shape=(len(lines),len((lines[0].split()))))
    col_indexes = set()
    fs = 0
    for i in range(len(lines)):
        features = lines[i].split()
        fs = len(features)
        for j in range(len(features)):
            cols[i].append(features[j])
            if features[j].isalpha():
                col_indexes.add(j)

    print 'cols = ', col_indexes

    for each in col_indexes:
        le = preprocessing.LabelEncoder()
        list_to_pass = []

        for i in range(len(lines)):
            list_to_pass.append(cols[i][each])

        le.fit(list_to_pass)
        print le.classes_
        fitted_list = le.transform(list_to_pass)

        for i in range(len(lines)):
            cols[i][each] = fitted_list[i]

    matrix = np.asanyarray(cols,dtype=float)
    return matrix

def partitionDataset2(matrix):
    indexes = range(matrix.shape[0])
    partition = int(len(matrix) * 0.7)

    training_indexes, testing_indexes = indexes[:partition], indexes[partition:]
    print 'train indexes = ', training_indexes
    training, testing = matrix[training_indexes,:], matrix[testing_indexes,:]
    # print len(training), len(testing)

    training_truths_index = len(training[0]) - 1
    training_truths = training[:,training_truths_index]
    print 'length of train truths = ',len(training_truths)
    training_matrix = training
    training = np.delete(training,training_truths_index,1)
    print 'length of training columns = ', len(training[0])
    training_truths_to_return = np.zeros(shape=(len(training_truths),1))
    for i in range(len(training_truths)):
        training_truths_to_return[i,0] = training_truths[i]

    testing_truths_index = len(testing[0]) - 1
    testing_truths = testing[:,testing_truths_index]
    print 'length of test truths = ',len(testing_truths)
    testing = np.delete(testing,testing_truths_index,1)
    print 'length of test columns = ', len(testing[0])
    testing_truths_to_return = np.zeros(shape=(len(testing_truths),1))
    for i in range(len(testing_truths)):
        testing_truths_to_return[i,0] = testing_truths[i]

    return training,training_truths_to_return,testing,testing_truths_to_return


def createTree(root, index_list, my_column, val, matrix, truths):
    if len(np.unique(truths))== 1:
        '''TODO:
            1.Make this a leaf node,
            2.Store its predicted decision value
        '''
        root = Node()
        root.is_leaf = True
        root.own_matrix = matrix
        decision_value = (np.unique(truths))[0]
        root.decision_value = decision_value
        return root
    else:
        root = Node()
        root.column_number = my_column
        root.val = val
        left_matrix = np.empty((0,matrix.shape[1]),float)
        left_truths = np.empty((0,1),float)
        right_matrix = np.empty((0,matrix.shape[1]),float)
        right_truths = np.empty((0,1),float)

        row_number = 0
        for each_row in matrix:
            if each_row[my_column] <= val:
                left_matrix = np.vstack((left_matrix,each_row))
                left_truths = np.vstack((left_truths,truths[row_number,0]))
            elif each_row[my_column] > val:
                right_matrix = np.vstack((right_matrix,each_row))
                right_truths = np.vstack((right_truths,truths[row_number,0]))
            row_number = row_number + 1

        lgains_list,lgain_values,lgains_index_list = calGain(left_matrix,left_truths)
        left_my_column = lgain_values.index(max(lgain_values))
        left_val = lgains_list[lgain_values.index(max(lgain_values))]
        root.left = createTree(None,index_list,left_my_column,left_val,left_matrix,left_truths)
        rgains_list,rgain_values,rgains_index_list = calGain(right_matrix,right_truths)
        right_my_column = rgain_values.index(max(rgain_values))
        right_val = rgains_list[rgain_values.index(max(rgain_values))]
        root.right = createTree(None,index_list,right_my_column,right_val,right_matrix,right_truths)
    return root

def tenCrossValidations(matrix,truths):
    avg_acc = 0.0
    kf = KFold(len(matrix), n_folds=10)
    for train, test in kf:
        training, testing, training_truths, testing_truths = matrix[train], matrix[test], truths[train], truths[test]

        print 'asasd',training_truths.shape
        print 'asasd',testing_truths.shape
        training_truths = np.reshape(training_truths,(len(training_truths),1))
        testing_truths= np.reshape(testing_truths,(len(testing_truths),1))


        gains_list,gain_values,gains_index_list = calGain(training,training_truths)
        my_column = gain_values.index(max(gain_values))
        val = gains_list[gain_values.index(max(gain_values))]
        root = createTree(None,[0],my_column,val,training,training_truths)
        print root.val
        predicted_label = dtpredict(root, testing, testing_truths)
        acc = 0
        for i in range(len(predicted_label)):
            if predicted_label[i] == testing_truths[i]:
               acc += 1
        print 'my acc = ',float(acc)/float(len(testing_truths))

        confusion_matrix = np.zeros((2,2))
        for i in range(0,len(testing_truths)):
            confusion_matrix[testing_truths[i][0]][predicted_label[i]] += 1
        recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
        precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
        f1 = (confusion_matrix[1][1] *2)/((confusion_matrix[1][1] *2)+confusion_matrix[0][1]+confusion_matrix[1][0])
        accuracy = 100*(confusion_matrix[1][1]+confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])

        print "Recall: ",recall
        print "Precision: ",precision
        print "F1 Measure: ",f1
        print "Accuracy: ",accuracy,"%"
        avg_acc += accuracy

    print 'avg acc = ',float(avg_acc)/float(10)
    return

if __name__ == '__main__':
    print 'for dataset 1..'
    matrix2 = loadDataset2('data/dataset1.txt')
    training,training_truths,testing,testing_truths = partitionDataset2(matrix2)
    # training,training_truths,testing,testing_truths = loadData('data/dataset1.txt')

    gains_list,gain_values,gains_index_list = calGain(training,training_truths)
    print 'gains list = ', gains_list
    print 'gain values',gain_values
    print 'gains indexes list = ',gains_index_list
    my_column = gain_values.index(max(gain_values))
    val = gains_list[gain_values.index(max(gain_values))]
    print 'my col, val',my_column,val
    root = createTree(None,[0],my_column,val,training,training_truths)
    print root.val
    predicted_label = dtpredict(root, testing, testing_truths)
    acc = 0
    for i in range(len(predicted_label)):
        if predicted_label[i] == testing_truths[i]:
           acc += 1
    print 'my acc = ',float(acc)/float(len(testing_truths))

    confusion_matrix = np.zeros((2,2))
    for i in range(0,len(testing_truths)):
        confusion_matrix[testing_truths[i][0]][predicted_label[i]] += 1
    recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
    precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
    f1 = (confusion_matrix[1][1] *2)/((confusion_matrix[1][1] *2)+confusion_matrix[0][1]+confusion_matrix[1][0])
    accuracy = 100*(confusion_matrix[1][1]+confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])

    print "Recall: ",recall
    print "Precision: ",precision
    print "F1 Measure: ",f1
    print "Accuracy: ",accuracy,"%"

    if str(raw_input('do you want to perform 10 cross validation')) == 'yes':
        matrix2 = loadDataset2('data/dataset1.txt')
        tenCrossValidations(matrix2[:,range(len(matrix2[0])-1)],matrix2[:,len(matrix2[0])-1])
        exit()
    else:
        pass


    '''For dataset2'''
    print 'for dataset 2..'
    matrix2 = loadDataset2('data/dataset2.txt')
    training,training_truths,testing,testing_truths = partitionDataset2(matrix2)
    # training,training_truths,testing,testing_truths = loadData('data/dataset1.txt')

    gains_list,gain_values,gains_index_list = calGain(training,training_truths)
    print 'gains list = ', gains_list
    print 'gain values',gain_values
    print 'gains indexes list = ',gains_index_list
    my_column = gain_values.index(max(gain_values))
    val = gains_list[gain_values.index(max(gain_values))]
    print 'my col, val',my_column,val
    root = createTree(None,[0],my_column,val,training,training_truths)
    print root.val
    predicted_label = dtpredict(root, testing, testing_truths)
    acc = 0
    for i in range(len(predicted_label)):
        if predicted_label[i] == testing_truths[i]:
           acc += 1
    print 'my acc = ',float(acc)/float(len(testing_truths))

    confusion_matrix = np.zeros((2,2))
    for i in range(0,len(testing_truths)):
        confusion_matrix[testing_truths[i][0]][predicted_label[i]] += 1
    recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
    precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
    f1 = (confusion_matrix[1][1] *2)/((confusion_matrix[1][1] *2)+confusion_matrix[0][1]+confusion_matrix[1][0])
    accuracy = 100*(confusion_matrix[1][1]+confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])

    print "Recall: ",recall
    print "Precision: ",precision
    print "F1 Measure: ",f1
    print "Accuracy: ",accuracy,"%"

    if str(raw_input('do you want to perform 10 cross validation')) == 'yes':
        matrix2 = loadDataset2('data/dataset2.txt')
        tenCrossValidations(matrix2[:,range(len(matrix2[0])-1)],matrix2[:,len(matrix2[0])-1])
        exit()
    else:
        pass

    exit()

    val = calculateColGains(training,training_truths)
    print 'split column = ',val.index(max(val))

    index_list = [ ]
    for i in range(training.shape[1]):
        index_list.insert(i,i)

    tree = DecisionTree()
    root = createTree(None,index_list,val.index(max(val)),training,training_truths)
    predicted_label = dtpredict(root, testing, testing_truths)
    # for i in range(0,testing_truths.shape[0]):
    #     print testing_truths[i], predicted_label[i]
    confusion_matrix = np.zeros((2,2))
    for i in range(0,len(testing_truths)):
        #print test_label[i][0], predicted_label[i][0]
        print testing_truths[i][0],predicted_label[i]
        confusion_matrix[testing_truths[i][0]][predicted_label[i]] += 1
    #print confusion_matrix
    recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
    precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
    f1 = (confusion_matrix[1][1] *2)/((confusion_matrix[1][1] *2)+confusion_matrix[0][1]+confusion_matrix[1][0])
    accuracy = 100*(confusion_matrix[1][1]+confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])

    print "Recall: ",recall
    print "Precision: ",precision
    print "F1 Measure: ",f1
    print "Accuracy: ",accuracy,"%"
    print 'global list = ',global_list
    # print 'first split value = ',col_num

    exit()