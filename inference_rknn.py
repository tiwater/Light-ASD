import os, tqdm, torch, glob, cv2, numpy, math, python_speech_features
import onnxruntime as ort
import numpy as np
import os
from scipy.io import wavfile
from rknnlite.api import RKNNLite

pycropPath = "demo/test/pycrop"

def evaluate_network(files):
# Initialize RKNN model
  lightAsd = RKNNLite()
  lightAsd.load_rknn('./models/lightASD.rknn')
  lightAsd.init_runtime()

  lossAV = RKNNLite()
  lossAV.load_rknn('./models/lossAV.rknn')
  lossAV.init_runtime()

  allScores = []
  # durationSet = {1,2,4,6} # To make the result more reliable
  durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
  for file in tqdm.tqdm(files, total = len(files)):
    fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
    _, audio = wavfile.read(os.path.join(pycropPath, fileName + '.wav'))
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
    video = cv2.VideoCapture(os.path.join(pycropPath, fileName + '.avi'))
    videoFeature = []
    while video.isOpened():
      ret, frames = video.read()
      if ret == True:
        face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224,224))
        face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
        videoFeature.append(face)
      else:
        break
    video.release()
    videoFeature = numpy.array(videoFeature)
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    allScore = [] # Evaluation use model
    for duration in durationSet:
      batchSize = int(math.ceil(length / duration))
      scores = []
      with torch.no_grad():
        for i in range(batchSize):
          inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).numpy()
          inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).numpy()
          
          # Transpose visual input to match NHWC format
          inputV = inputV.transpose(0, 2, 3, 1)  # from NCHW to NHWC
          out = lightAsd.inference(inputs=[inputA, inputV])[0]
          
          # print(out)
          # print(out.shape)

          # Get inference for the target image
          score = lossAV.inference(inputs=[out])

          # print(score)
          scores.extend(score)
      allScore.append(scores)
    allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
    allScores.append(allScore)	

    lightAsd.release()
  return allScores

# Active Speaker Detection
files = glob.glob("%s/*.avi"%pycropPath)
files.sort()
scores=evaluate_network(files)
# print(scores)