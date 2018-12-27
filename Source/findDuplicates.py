#!/usr/bin/env python
import sys
import os
import hashlib
from Source.readFile import *
from scipy.spatial import distance

def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filename, first_chunk_only=False, hash=hashlib.sha1):
    hashobj = hash()
    file_object = open(filename, 'rb')

    if first_chunk_only:
        hashobj.update(file_object.read(1024))
    else:
        for chunk in chunk_reader(file_object):
            hashobj.update(chunk)
    hashed = hashobj.digest()

    file_object.close()
    return hashed


def check_for_duplicates(paths, hash=hashlib.sha1):
    hashes_by_size = {}
    hashes_on_1k = {}
    hashes_full = {}
    duplicates = []
    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                try:
                    file_size = os.path.getsize(full_path)
                except (OSError,):
                    # not accessible (permissions, etc) - pass on
                    pass

                duplicate = hashes_by_size.get(file_size)

                if duplicate:
                    hashes_by_size[file_size].append(full_path)
                else:
                    hashes_by_size[file_size] = []  # create the list for this file size
                    hashes_by_size[file_size].append(full_path)

    # For all files with the same file size, get their hash on the 1st 1024 bytes
    for __, files in hashes_by_size.items():
        if len(files) < 2:
            continue  # this file size is unique, no need to spend cpy cycles on it

        for filename in files:
            small_hash = get_hash(filename, first_chunk_only=True)

            duplicate = hashes_on_1k.get(small_hash)
            if duplicate:
                hashes_on_1k[small_hash].append(filename)
            else:
                hashes_on_1k[small_hash] = []  # create the list for this 1k hash
                hashes_on_1k[small_hash].append(filename)

    # For all files with the hash on the 1st 1024 bytes, get their hash on the full file - collisions will be duplicates
    for __, files in hashes_on_1k.items():
        if len(files) < 2:
            continue  # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

        for filename in files:
            full_hash = get_hash(filename, first_chunk_only=False)

            duplicate = hashes_full.get(full_hash)
            if duplicate:
                duplicates.append((filename, duplicate))
                print("Duplicate found: %s and %s" % (filename, duplicate))
            else:
                hashes_full[full_hash] = filename
    # Writing to the file.
    mfile = open('../Docs/duplicates.txt',
                  mode='w')
    for row in duplicates:
        mfile.write("Duplicate found: %s and %s \n" % (row[0], row[1]))
    mfile.close()

    # Deleting duplicates.
    means = []
    for i in range (1,9):
        means.append(findMean(i))
    genres = {"Metal": 1,"Rock": 2,"Jazz":3,"Rap":4,"Electronic":5,"Pop":6,"Soundtrack":7,"Classical":8}
    for row in duplicates:
        dup1genre = row[0].split("/")[-3]
        dup2genre = row[1].split("/")[-3]
        song = readFile(row[0], 0)
        distance1 = distance.euclidean(song[1:],means[genres[dup1genre]-1])
        distance2 = distance.euclidean(song[1:],means[genres[dup2genre]-1])
        if distance1 > distance2:
            if os.path.isfile(row[0]):
                os.remove(row[0])
            else:
                os.remove(row[1])
        else:
            if os.path.isfile(row[1]):
                os.remove(row[1])
            else:
                os.remove(row[0])
    '''for row in duplicates:
        if os.path.isfile(row[1]):
            os.remove(row[1])
        else:
            os.remove(row[0])'''


if sys.argv[1:]:
    check_for_duplicates(sys.argv[1:])
else:
    print("Please pass the paths to check as parameters to the script")
