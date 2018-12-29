import numpy
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
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
    svdX = TruncatedSVD().fit_transform(X)
    svdTest = TruncatedSVD().fit_transform(numpy.array(test))
    clf.fit(svdX, Y)
    res = clf.predict(svdTest)
    score = 0
    testLabels = numpy.array(testLabels)
    for i in range(len(res)):
        if res[i] == testLabels[i]:
            score += 1
    # Plots the data - To be worked on
    plot_decision_regions(X=svdX,
                          y=Y,
                          clf=clf,
                         legend=2)
    plt.show()

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
def findMeanInPlaylist(playlist):
    count =0
    totals = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    for song in playlist:
        count+=1
        for i, feature in enumerate(song):
            totals[i] += feature
    res = [feature/count for feature in totals]
    return res
def calculateDist(song1,song2):
    dst = 0
    for i in range(len(song1)):
        dst += (song1[i]- song2[i])**2
    return dst
def getSongsInRange(distRange,nSongs,songs,meanValue):
    lst = []
    offset = 0
    for song in songs:
        dst = calculateDist(song,meanValue)
        if(dst > distRange[0] and dst < distRange[1] ):
            lst.append(song)
    if(len(lst) < nSongs):
        offset = nSongs - len(lst)
    sampled = random.sample(lst, min(len(lst),nSongs))
    sampled += random.sample(songs,offset)
    return sampled
def exp3Dist(allData,playlist):
    meanV = findMeanInPlaylist(playlist)
    minDist = 9999999999
    maxDist = 0
    for song in allData:
        dst = calculateDist(song,meanV)
        if(dst > maxDist):
            maxDist = dst
        if(dst < minDist):
            minDist =dst
    delta = (maxDist-minDist)/8
    distRange = (0,delta)
    nSongs = len(playlist)/ 7
    initNum = len(playlist)
    sampledSongs = []
    for i in range(0,7):
        initNum -= math.floor(nSongs)
        if i== 6 and initNum != 0:
            sampledSongs += getSongsInRange(distRange,math.floor(nSongs)+math.floor(initNum),allData,meanV)
            incrDel = (8 -i)*delta
            distRange = (distRange[0] + incrDel, distRange[1] + incrDel)
        else:
            sampledSongs += getSongsInRange(distRange, math.floor(nSongs), allData, meanV)
            distRange = (distRange[0]+delta,distRange[1]+delta)

    return sampledSongs
def exp4Dist(allData,playlist):
    meanV = findMeanInPlaylist(playlist)
    minDist = 9999999999
    maxDist = 0
    for song in allData:
        dst = calculateDist(song,meanV)
        if(dst > maxDist):
            maxDist = dst
        if(dst < minDist):
            minDist =dst
    delta = (maxDist-minDist)/2
    distRange = (delta,maxDist)
    sampledSongs = getSongsInRange(distRange,math.floor(len(playlist)),allData,meanV)
    return sampledSongs
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
        findMeanInPlaylist(A)
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
    
    #Experiment 3: Select train data by distance
    accTotal = 0
    cnt = 0
    for playlist in allData:
        A, B = split_list(playlist)
        train = A[:]
        labels = [1] * len(A)
        train += exp3Dist(songs,A)
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        test += random.sample(songs, len(B))
        testLabels += [0] * len(B)
        acc = trainModel(train, labels, test, testLabels)
        print("Playlist read. Accuracy: ", acc * 100, "%")
        accTotal += acc
        cnt += 1
    print("Experiment done. Overall accuracy: ", 100 * accTotal / cnt, '%')
    # Experiment 4: Get farthest distances
    accTotal = 0
    cnt = 0
    for playlist in allData:
        A, B = split_list(playlist)
        train = A[:]
        labels = [1] * len(A)
        train += exp4Dist(songs, A)
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        test += random.sample(songs, len(B))
        testLabels += [0] * len(B)
        acc = trainModel(train, labels, test, testLabels)
        print("Playlist read. Accuracy: ", acc * 100, "%")
        accTotal += acc
        cnt += 1
    print("Experiment done. Overall accuracy: ", 100 * accTotal / cnt, '%')
