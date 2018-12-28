import numpy
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
# Change this to 'from readFile import *' if you are using Python 2.*
from Source.readFile import *
# Returns the train and test data as well as labels for train data.
def getData():
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [item for sublist in l for item in sublist]
    train = []
    test= []
    labels = []
    for i in range(1,9):
        A,B = split_list(readGenreDirectory("../Dataset/"+ genres[i]))
        A = flatten(A)
        train.append(A)
        labels+= [i]* len(A)
        test.append(flatten(B))
    return (flatten(train),flatten(test),labels)
# Trains the model using SciKit Learn SVC
def trainModel():
    playlists = getData()
    clf = svm.SVC(gamma='scale')
    value = 1
    width = 1
    X = numpy.array(playlists[0])
    Y = numpy.array(playlists[2])
    clf.fit(X,Y)
    #Plots the data - To be worked on
    plot_decision_regions(X=X,
                          y=Y,
                          clf=clf,
                          filler_feature_values={2: value, 3: value, 4: value ,5 : value , 6:value,7:value, 8:value,9:value,10:value,11:value},
                          filler_feature_ranges={2: width, 3: width, 4: width,5 : width , 6:width,7:width, 8:width,9:width,10:width,11:width},
                          legend=2)
    plt.show()
#Utility function
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
if __name__ == "__main__":
    trainModel()
