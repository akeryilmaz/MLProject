import os
import numpy

def readFile(path, genreTag):
    '''reads a song file of given path(string), returns the feateures and genre tag in a list where the first element is tag
    CAUTION!!! It maps year, loudness and tempo to 0-1 range'''
    songFile = open(path,"r")
    result = []
    result.append(genreTag)
    for line in songFile:
        if line.split(':')[0] == "name":
            continue
        elif line.split(':')[0] == "release_year":
            result.append((float(line.split(':')[1][:-1])-1900)/119)
            continue
        elif line.split(':')[0] == "loudness":
            result.append((float(line.split(':')[1][:-1])+60)/60)
            continue
        elif line.split(':')[0] == "tempo":
            result.append((float(line.split(':')[1][:-1]))/212.349)
            continue
        result.append(float(line.split(':')[1][:-1]))
    return result

def readPlaylistDirectory(path, genreTag):
    '''reads song files in the playlist folder of given path(string), returns the song feature lists in a list'''
    result = []
    for root, subDirectories, files in os.walk(path):
        for songFile in files:
            result.append(readFile(path + "/" +  songFile, genreTag))
    return result


def readGenreDirectory(path):
    '''reads song files in the genre folder of given path(string), tags the songs returns the song feature lists in a list of list'''
    result = []
    genres = {"Metal": 1,"Rock": 2,"Jazz":3,"Rap":4,"Electronic":5,"Pop":6,"Soundtrack":7,"Classical":8}
    currentGenre = path.split("/")[-1]
    tag = genres[currentGenre]
    for root, subDirectories, files in os.walk(path):
        for subDirectory in subDirectories:
            result.append(readPlaylistDirectory(path + "/" + subDirectory, tag))
    return result

def findMean(tag):
    ''' Returns the mean values of the features in a given genre '''
    genres = {1: "Metal",2: "Rock",3:"Jazz",4:"Rap",5:"Electronic",6:"Pop",7:"Soundtrack",8:"Classical"}
    for root, subDirectories, files in os.walk(os.path.abspath("../Dataset")):
        for subDirectory in subDirectories:
            if subDirectory == genres[tag]:
                result = readGenreDirectory("../Dataset/" + subDirectory)
        break
    totals = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    count = 0
    for playlist in result:
        for song in playlist:
            count += 1
            for i, feature in enumerate(song[1:]):
                totals[i] += feature
    return [feature/count for feature in totals]

def getNearest(lst):
        dstDict = {lst[i][0]: [] for i in range(len(lst))}
        ind = 0
        for ftrs in lst:
            for otherIndex, otherftrs in enumerate(lst):
                dst = 0
                if ftrs[1] == otherftrs[1]:
                    continue
                for i in range(1, len(otherftrs[1])):
                    dst += (otherftrs[1][i] - ftrs[1][i]) ** 2
                dstDict[ftrs[0]].append({otherftrs[0] : dst})
            ind += 1

        return dstDict
if __name__ == "__main__":
    '''Find the means'''
    lst = []
    genres = {1: "Metal",2: "Rock",3:"Jazz",4:"Rap",5:"Electronic",6:"Pop",7:"Soundtrack",8:"Classical"}

    for i in range (1,9):
        lst.append((genres[i],findMean(i)))
    getNearest(lst)