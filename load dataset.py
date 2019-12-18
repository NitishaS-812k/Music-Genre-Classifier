"""
this file loads all the files in the dataset and extracts their features using librosa
the extracted features are placed in a csv file called data.csv 
"""
#required imports
import os
import numpy as np
import librosa
from tinytag import TinyTag
import csv
import warnings
warnings.filterwarnings('ignore') #to ignore soundfile warning

#location of audio data files
AUDIO_DIR = 'data/fma_small'

def get_trackid_from_directory(AUDIO_DIR):
#this function returns a list of all track_ids(names of the files)
    track_ids = []
    for _,dirnames, files in os.walk(AUDIO_DIR):
        if dirnames == []:
            track_ids.extend(int(file[:-4])for file in files)
    return track_ids

def get_audio_path(AUDIO_DIR,track_id):
#this function is called for each track_id and it returns the path of the file as a string
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(AUDIO_DIR,tid_str[:3], tid_str + '.mp3')

#this list defines the headers in the csv file
x = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    x += f' mfcc{i}'
x += ' label'
header = x.split()


def getGenre(track_id):
#this function is called for each file and it gets the genre as specified in metadata of the file
    tag = TinyTag.get(get_audio_path(AUDIO_DIR,track_id))
    if tag.genre == None:
        return str("None")
    return "".join(tag.genre.split()) 

def create_features_csv():
#this function opens a .csv file and begins writing the data for each .mp3 file in it
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)     #writing the headers into the file
    tid = get_trackid_from_directory(AUDIO_DIR)  
    for track_id in tid:
        filename = get_audio_path(AUDIO_DIR,track_id)
        try:
            y,sr = librosa.load(filename)      #librosa.load reads the file
            chroma_stft = librosa.feature.chroma_stft(y=y,sr=sr)    
            spec_cent = librosa.feature.spectral_centroid(y=y,sr = sr) 
            #spectral centroid indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound
            spec_bw = librosa.feature.spectral_bandwidth(y=y,sr =sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr =sr)
            #rolloff is a measure of the shape of the signal.
            zcr = librosa.feature.zero_crossing_rate(y=y) 
            #The zero crossing rate is the rate of sign-changes along a signal
            mfcc = librosa.feature.mfcc(y=y,sr =sr)
            #mfcc concisely describe the overall shape of a spectral envelope
            to_write = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '    
            for i in mfcc:
                to_write += f'{np.mean(i)} '
            to_write += f'{getGenre(track_id)}'
            file = open('data.csv', 'a', newline= '')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_write.split())
            print(filename)
        except:
            continue
if __name__ == "__main__":
    create_features_csv()
    print("over")