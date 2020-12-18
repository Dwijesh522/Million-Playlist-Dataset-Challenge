"""
    This file takes path to a directory. The directory is
    expected to have mpd.slice.*.json files. The program
    generates on-disk inverted index of the form:
        { track: {playlist : freq} }
        { pid: last_modified}
"""

import json
import sys
from os import listdir
from os.path import isfile, join

def error(l):
    if (len(l) != 2):
        print("ERROR: command line arg expected")
        exit(1)

def dumpToJson(dt, filename):
    """ dump dictionary dt to json file"""
    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)

def dump_invidx(slices_path):
    slices = [join(slices_path, f) for f in listdir(slices_path) if isfile(join(slices_path, f))]
    invidx = {}
    modifications = {}
    for f in slices:
        print(f)
        with open(f, 'r') as file_obj:
            data = json.load(file_obj)
            # ignoring slice, version, description, licence, generated_on keys
            # only keeping the playlists
            playlists = data['playlists']
            # interested in pid, modified_at, track_uri
            for playlist in playlists:
                pid = playlist['pid']
                modified_at = playlist['modified_at']
                modifications[pid] = modified_at
                tracks = playlist['tracks'] # list of dicts
                for track in tracks:
                    track_uri = track['track_uri']
                    if track_uri in invidx:
                        if pid in invidx[track_uri]:
                            invidx[track_uri][pid] += 1
                        else:
                            invidx[track_uri][pid] = 1
                    else:
                        invidx[track_uri] = {pid: 1}
    # writting to file
    dumpToJson(invidx, "invidx.json")
    dumpToJson(modifications, "modifications.json")

if __name__ == "__main__":
    error(sys.argv)
    slices_path = sys.argv[1]
    dump_invidx(slices_path)
