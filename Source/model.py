from readFile import *
from ExperimentSuite import *
import numpy
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from scipy.spatial import distance
import math
import random
# Change this to 'from readFile import *' if you are using Python 2.*
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    playlist, namePlaylist = readPlaylistDirectory("../Dataset/Rock/RockClassics", "Rock")
    songs = []
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [sublist for sublist in l]
    allData = []
    allNames = []
    for i in range(1, 9):
        genreDirectoryTuple = readGenreDirectory("../Dataset/" + genres[i])
        allNames += genreDirectoryTuple[1]
        allData += genreDirectoryTuple[0]
    names = []
    for j, playlist1 in enumerate(allData):
        names += flatten(allNames[j])
        songs += flatten(playlist1[1:])
    train = flatten(playlist[1:])
    numberOfSongs = len(train)
    labels = [1] * numberOfSongs
    train += exp4Dist(songs, train)
    labels += [0] * numberOfSongs
    X = numpy.array(train)
    Y = numpy.array(labels)
    clf = svm.SVC(gamma='scale', kernel='rbf', C=1.0, probability=True)
    clf.fit(X, Y)
    results = clf.predict_proba(songs)
    probabilities = []
    for result in results:
        probabilities.append(result[1])
    recommendations = []
    for i in range (0,10):
        currentIndex = probabilities.index(max(probabilities))
        recommendations.append(names[currentIndex])
        del probabilities[currentIndex]
        del names[currentIndex]
    print recommendations