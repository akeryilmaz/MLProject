import os

def readFile(path):
    '''reads a file of given path(string), returns the values in a list
    CAUTION!!! It maps year, loudness and tempo to 0-1 range'''
    songFile = open(path,"r")
    result = []
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
        
def readPlaylistDirectory(path):
    '''reads files in the folder of given path(string), returns the values in a list'''
    result = []
    for root, subDirectories, files in os.walk(path):
        for songFile in files:
            result.append(readFile(path + "/" +  songFile))
    return result


def readGenreDirectory(path):
    result = []
    for root, subDirectories, files in os.walk(path):
        for subDirectory in subDirectories:
            result.append(readPlaylistDirectory(path + "/" + subDirectory))
    return result

if __name__ == "__main__":
    '''Find the minimum and maximum year of the songs'''
    minYear = float(9999999)
    maxYear = float(0)
    allFeatures = []
    for root, subDirectories, files in os.walk(os.path.abspath("/home/akeryilmaz/Desktop/Ceng499/project/Dataset")):
        for subDirectory in subDirectories:
            allFeatures.append(readGenreDirectory("/home/akeryilmaz/Desktop/Ceng499/project/Dataset/" + subDirectory))
    for genre in allFeatures:
        for playlist in genre:
            for song in playlist:
                year = song[0]
                if year < minYear:
                    minYear = year
                if year > maxYear:
                    maxYear = year
    print minYear, maxYear