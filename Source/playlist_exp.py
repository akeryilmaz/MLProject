import numpy
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
import random
# Change this to 'from readFile import *' if you are using Python 2.*
from readFile import *
# Returns the train and test data as well as labels for train data.
# Trains the model using SciKit Learn SVC
def trainModel():
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [item for sublist in l for item in sublist]
    allData = []
    for i in range(1,9):
        allData += readGenreDirectory("../Dataset/"+ genres[i])
    songs = flatten(allData)
    labels = []
    testLabels = []
    for playlist in allData:
        A,B = split_list(playlist)
        train = A[:]
        labels = [1]* len(A)
        train += random.sample(songs, len(A))
        labels += [0]* len(A)
        test = B[:]
        testLabels = [1]*len(B)
        test += random.sample(songs, len(B))
        testLabels += [0]*len(B)
        clf = svm.SVC(gamma='scale',kernel='rbf', C=1.0)
        value = 1
        width = 1
        X = numpy.array(train)
        Y = numpy.array(labels)
        clf.fit(X,Y)
        res = clf.predict(test)
        score = 0
        testLabels = numpy.array(testLabels)
        print res
        for i in  range(len(res)):
            if res[i] == testLabels[i]:
                score+=1
        print score / len(test)
        #Plots the data - To be worked on
        plot_decision_regions(X=X,
                            y=Y,
                            clf=clf,
                            filler_feature_values={2: value, 3: value, 4: value ,5 : value , 6:value,7:value, 8:value,9:value,10:value},
                            filler_feature_ranges={2: width, 3: width, 4: width,5 : width , 6:width,7:width, 8:width,9:width,10:width},
                            legend=2)
        plt.show()
#Utility function
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
if __name__ == "__main__":
    trainModel()
