"""
    This file dumps pairwise similarity values on disk.
    command line arguments:
    1) topm.npy             : {qpid : numpy.ndarray(pid)}
    2) <pid_vectors_dir>    : path to tfidf vector representation of pids dir
    3) challenge_set_filtered.npy : {qpid : numpy.ndarray(track_uri)}

    output files:
    1) similarities.json {qpid : {pid : tfidf}}
"""

import json
import sys
import numpy as np
import math

def error(args):
    if (len(args) != 4):
        print("ERROR: command line args expected")
        exit(1)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def loadFromNpy(filename):
    """ returns the dictionary: {pid : numpy.ndarray(track_uri)}
        numpy array data type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

def dumpToJson(dt, filename):
    """ dump dictionary dt to json file"""
    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)

def dumpSimilarity(topm, tfidf_path, challenge_set):
    similarity = {qpid : {} for qpid in topm}

    # fetching tfidf of all slices
    iterations = 2 # must divide 1000
    slices_per_iterations = 1000//iterations
    for i in range(iterations):
        start = i*slices_per_iterations
        end = (i+1)*slices_per_iterations # not inclusive
        tfidf = np.empty(end-start, 'object')
        for counter in range(start, end, 1):
            print(counter, " tfidf fetched")
            search_filename = tfidf_path + "/" + str(counter*1000) + "_pid_vectors.json"
            tfidf[counter-start] = fetchDict(search_filename) # [ {pid : {tracks : tfidf}} ]
        
        j = 0
        for qpid in topm: # type(qpid) = int
            qtracks = challenge_set[qpid]
            for pid in topm[qpid]: # type(pid) = int
                count = pid // 1000
                if (count < start) or (count >= end) : continue
                modulus = math.sqrt(sum(v*v for v in tfidf[count-start][str(pid)].values()))
                dot_prod = sum(tfidf[count-start][str(pid)][qtrack_uri] for qtrack_uri in qtracks if qtrack_uri in tfidf[count-start][str(pid)])
                similarity[qpid][str(pid)] = dot_prod/modulus
            print(j, " qpids done...")
            j += 1
    dumpToJson(similarity, "pairwise_similarity.json")

if __name__ == "__main__":
    error(sys.argv)
    topm = loadFromNpy(sys.argv[1])
    tfidf_path = sys.argv[2]
    challenge_set = loadFromNpy(sys.argv[3])
    print("reading data-structures to memory done...")

    dumpSimilarity(topm, tfidf_path, challenge_set)
