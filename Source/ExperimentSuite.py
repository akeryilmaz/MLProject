import numpy
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
import math
import random
# Change this to 'from readFile import *' if you are using Python 2.*
from readFile import *
# Returns the train and test data as well as labels for train data.
# Trains the model using SciKit Learn SVC
def trainModel(train,labels,test,testLabels):
    clf = svm.SVC(gamma='scale', kernel='rbf', C=1.0)
    value = 1
    width = 1
    X = numpy.array(train)
    Y = numpy.array(labels)
    clf.fit(X, Y)
    res = clf.predict(test)
    score = 0
    testLabels = numpy.array(testLabels)
    for i in range(len(res)):
        if res[i] == testLabels[i]:
            score += 1
    # Plots the data - To be worked on
    '''
    plot_decision_regions(X=X,
                          y=Y,
                          clf=clf,
                          filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value, 8: value,
                                                 9: value, 10: value},
                          filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width, 8: width,
                                                 9: width, 10: width},
                          legend=2)
    plt.show()
    '''
    return (score/len(test))
#Utility function
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
def randomNForGenre(number):
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [item for sublist in l for item in sublist]
    sampleNum = math.floor(number / len(genres.keys()))
    sampledGenre = []
    for i in range(1, 9):
        number -= sampleNum
        flat = flatten(readGenreDirectory("../Dataset/" + genres[i]))
        if(i == 8 and number != 0):
            sampledGenre += random.sample(flat,sampleNum+number)
        else:
            sampledGenre += random.sample(flat, sampleNum)

    return sampledGenre
if __name__ == "__main__":
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [item for sublist in l for item in sublist]
    allData = []
    for i in range(1, 9):
        allData += readGenreDirectory("../Dataset/" + genres[i])
    songs = flatten(allData)
    #Experiment 1: Randomize
    accTotal = 0
    cnt = 0
    for playlist in allData:
        A, B = split_list(playlist)
        train = A[:]
        labels = [1] * len(A)
        train += random.sample(songs, len(A))
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        test += random.sample(songs, len(B))
        testLabels += [0] * len(B)
        acc = trainModel(train,labels,test,testLabels)
        print("Playlist read. Accuracy: ", acc * 100, "%")
        accTotal +=acc
        cnt+=1
    print("Experiment done. Overall accuracy: ", 100 * accTotal / cnt, '%')
    #Experiment 2: Equal for each genre
    accTotal = 0
    cnt = 0
    for playlist in allData:
        A, B = split_list(playlist)
        train = A[:]
        labels = [1] * len(A)
        train += randomNForGenre(len(A))
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        test += random.sample(songs, len(B))
        testLabels += [0] * len(B)
        acc = trainModel(train,labels,test,testLabels)
        print("Playlist read. Accuracy: ",acc*100, "%")
        accTotal += acc
        cnt += 1
    print("Experiment done. Overall accuracy: ", 100*accTotal / cnt,'%')
