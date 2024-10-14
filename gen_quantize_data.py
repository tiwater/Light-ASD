import os, tqdm, torch, glob, cv2, numpy, math, python_speech_features
import onnxruntime as ort
import numpy as np
import os
from scipy.io import wavfile
import time

# Exec Columbia_test.py first to get the pycrop data

pycropPath = "demo/test/pycrop"

def gen_data(files):
  # durationSet = {1,2,4,6} # To make the result more reliable
  # durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
  durationSet = {6} # Must be the max value allowd in dynamic input
  dataset_path = 'data/dataset.txt'
  with open(dataset_path, 'w') as f:  # Open dataset file for writing.
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(pycropPath, fileName + '.avi'))
        videoFeature = []
        
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        
        video.release()
        videoFeature = np.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []  # Evaluation use model
        
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100, :]).unsqueeze(0).numpy()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25, :, :]).unsqueeze(0).numpy()
                    
                    # Check for required dimensions
                    if inputA.shape[1] != duration * 100 or inputV.shape[1] != duration * 25:
                        continue
                    # Save each round's inputA and inputV respectively as audio_{}.npy and video_{}.npy
                    audio_filename = f'data/audio_{fileName}_{duration}_{i}.npy'
                    video_filename = f'data/video_{fileName}_{duration}_{i}.npy'
                    np.save(audio_filename, inputA)
                    np.save(video_filename, inputV)
                    
                    # Save filenames to data/dataset.txt with both filenames in the same line separated by a space
                    f.write(f"{audio_filename} {video_filename}\n")

    return

# Active Speaker Detection
files = glob.glob("%s/*.avi"%pycropPath)
files.sort()
scores=gen_data(files)
# print(scores)