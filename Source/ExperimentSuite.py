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
    return (float(score)/len(test))
#Utility function
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
def randomNForGenre(number):
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [sublist for sublist in l]
    sampleNum = int(math.floor(number / len(genres.keys())))
    sampledGenre = []
    for i in range(1, 9):
        number -= sampleNum
        genre = readGenreDirectory("../Dataset/" + genres[i])
        flat = []
        for playlist in genre:
            flat += flatten(playlist[1:])
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
            sampledSongs += getSongsInRange(distRange, int(math.floor(nSongs)+math.floor(initNum)),allData,meanV)
            incrDel = (8 -i)*delta
            distRange = (distRange[0] + incrDel, distRange[1] + incrDel)
        else:
            sampledSongs += getSongsInRange(distRange, int(math.floor(nSongs)), allData, meanV)
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
    sampledSongs = getSongsInRange(distRange,int(math.floor(len(playlist))),allData,meanV)
    return sampledSongs
if __name__ == "__main__":
    genres = {1: "Metal", 2: "Rock", 3: "Jazz", 4: "Rap", 5: "Electronic", 6: "Pop", 7: "Soundtrack", 8: "Classical"}
    flatten = lambda l: [sublist for sublist in l]
    allData = []
    for i in range(1, 9):
        allData += readGenreDirectory("../Dataset/" + genres[i])
    #Experiment 1: Randomize
    accTotal = 0
    cnt = 0
    for i, playlist in enumerate(allData):
        A, B = split_list(playlist[1:])
        findMeanInPlaylist(A)
        train = A[:]
        labels = [1] * len(A)
        songs = []
        for j, playlist1 in enumerate(allData):
            if j != i:
                songs += flatten(playlist1[1:])
        trainSelected = []
        for i in range (0,len(A)):
            while True:
                newSong = random.randint(0,len(songs)-1)
                if newSong not in trainSelected:
                    trainSelected.append(newSong)
                    break
        for selectedSong in trainSelected:
            train.append(songs[selectedSong])
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        testSelected = []
        for i in range (0,len(B)):
            while True:
                newSong = random.randint(0,len(songs)-1)
                if newSong not in testSelected and newSong not in trainSelected :
                    testSelected.append(newSong)
                    break
        for selectedSong in testSelected:
            test.append(songs[selectedSong])
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
        A, B = split_list(playlist[1:])
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
        A, B = split_list(playlist[1:])
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
        A, B = split_list(playlist[1:])
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
    lst = []
    for i in range (1,9):
        lst.append((genres[i],findMean(i)))
    genreDistanceDict = getNearest(lst)
    # Experiment 5: select train data randmly from nearest 3 genres
    accTotal = 0
    cnt = 0
    for i, playlist in enumerate(allData):
        currentGenre = playlist[0]
        A, B = split_list(playlist[1:])
        train = A[:]
        labels = [1] * len(A)
        top3GenresPlaylists = []
        for otherGenre, distance in genreDistanceDict[currentGenre]:
            top3GenresPlaylists += readGenreDirectory("../Dataset/"+otherGenre)
        top3genreSongs = []
        for j, playlist1 in enumerate(allData):
            if j != i and currentGenre == playlist1[0]:
                top3genreSongs += flatten(playlist1[1:])
        for playlist1 in top3GenresPlaylists:
            top3genreSongs += flatten(playlist1[1:])
        trainSelected = []
        for i in range (0,len(A)):
            while True:
                newSong = random.randint(0,len(top3genreSongs)-1)
                if newSong not in trainSelected:
                    trainSelected.append(newSong)
                    break
        for selectedSong in trainSelected:
            train.append(top3genreSongs[selectedSong])
        labels += [0] * len(A)
        test = B[:]
        testLabels = [1] * len(B)
        allSongs = []
        for j, playlist1 in enumerate(allData):
            if j != i:
                allSongs += flatten(playlist1[1:])
        testSelected = []
        for i in range (0,len(B)):
            while True:
                newSong = random.randint(0,len(allSongs)-1)
                if newSong not in testSelected:
                    testSelected.append(newSong)
                    break
        for selectedSong in testSelected:
            test.append(allSongs[selectedSong])
        testLabels += [0] * len(B)
        acc = trainModel(train,labels,test,testLabels)
        print("Playlist read. Accuracy: ", acc * 100, "%")
        accTotal +=acc
        cnt+=1
    print("Experiment done. Overall accuracy: ", 100 * accTotal / cnt, '%')