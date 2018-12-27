import os

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

if __name__ == "__main__":
    '''Find the minimum and maximum year of the songs'''
    minYear = 9999999
    maxYear = -999999
    allFeatures = []
    for root, subDirectories, files in os.walk(os.path.abspath("../Dataset")):
        for subDirectory in subDirectories:
            allFeatures.append(readGenreDirectory("../Dataset/" + subDirectory))
        break
    for genre in allFeatures:
        for playlist in genre:
            for song in playlist:
                year = song[1]
                if year < minYear:
                    minYear = year
                if year > maxYear:
                    maxYear = year
    print(minYear, maxYear)