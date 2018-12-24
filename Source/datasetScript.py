import spotipy 
import json
import string
import os

sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
cid ="4f79378abd7843f4b8b18273010e58f2" 
secret = "9e4e97be9bae42dfaf9dbfe75a20c1aa" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 

playlistFile = open("../Docs/playlists.txt","r")
genres = {"1": "Metal","2": "Rock","3":"Jazz","4":"Rap","5":"Electronic","6":"Pop","7":"Soundtrack","8":"Classical"}
for line in playlistFile:
    if line[0] in genres.keys():
        currentGenre = line[0]
    else:
        username = line.split(':')[2]
        playlist_id = line.split(':')[4][:-1]
        print username, playlist_id
        playlistInfo = sp.user_playlist(username, playlist_id)
        songs = playlistInfo["tracks"]["items"]
        deletePuncs = {ord(char): None for char in string.punctuation}
        playlistName = playlistInfo["name"].translate(deletePuncs)
        playlistName = playlistName.replace(" ", "")
        for i in range(len(songs)):
            print i
            try:
                track = sp.track(str(songs[i]["track"]["id"]))
                features = sp.audio_features(str(songs[i]["track"]["id"]))
                name = songs[i]["track"]["name"].translate(deletePuncs)
                name = name.replace(" ", "")
                path = "../Dataset/"+genres[currentGenre]+"/"+ playlistName + "/" +  name +".txt"
                directory = os.path.dirname(path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                outputFile = open(path , "w")
                outputFile.write("release_date_precision:" + str(track["album"]["release_date_precision"])+ "\n")
                outputFile.write("release_date:" + str(track["album"]["release_date"] + "\n"))
                outputFile.write("danceability:" + str(features[0]["danceability"])+ "\n")
                outputFile.write("energy:" + str(features[0]["energy"])+ "\n")
                outputFile.write("loudness:" + str(features[0]["loudness"])+ "\n")
                outputFile.write("mode:" + str(features[0]["mode"])+ "\n")
                outputFile.write("speechiness:" + str(features[0]["speechiness"])+ "\n")
                outputFile.write("acousticness:" + str(features[0]["acousticness"])+ "\n")
                outputFile.write("instrumentalness:" + str(features[0]["instrumentalness"])+ "\n")
                outputFile.write("liveness:" + str(features[0]["liveness"])+ "\n")
                outputFile.write("valence:" + str(features[0]["valence"])+ "\n")
                outputFile.write("tempo:" + str(features[0]["tempo"])+ "\n")
                outputFile.close()
            except:
                print "Something went wrong!"
playlistFile.close() 