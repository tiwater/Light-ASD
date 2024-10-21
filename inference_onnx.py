import os
import numpy as np
import glob
import cv2
from scipy.io import wavfile
import python_speech_features
import tqdm
import onnxruntime as ort
import time

pycropPath = "demo/test/pycrop"

def evaluate_network(files):
    providers = ['CPUExecutionProvider']
    model = ort.InferenceSession('./models/lightASD.onnx', providers=providers)
    lossAV = ort.InferenceSession('./models/lossAV.onnx', providers=providers)

    allScores = []
    # Use this line can get more reliable result
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}

    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(os.path.basename(file))[0] 

        # Load audio and video
        _, audio = wavfile.read(os.path.join(pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        video = cv2.VideoCapture(os.path.join(pycropPath, fileName + '.avi'))
        videoFeature = []

        while video.isOpened():
            ret, frames = video.read()
            if not ret:
                break
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[56:168, 56:168]  # Crop centrally without additional calculations
            videoFeature.append(face)

        video.release()
        videoFeature = np.array(videoFeature)

        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100))]
        videoFeature = videoFeature[:int(round(length * 25))]

        allScore = []  # Evaluation use model
        for duration in durationSet:
            batchSize = int(np.ceil(length / duration))
            scores = []
            
            for i in range(batchSize):
                inputA = audioFeature[i * duration * 100:(i+1) * duration * 100, :].astype(np.float32)[np.newaxis]
                inputV = videoFeature[i * duration * 25:(i+1) * duration * 25, :, :].astype(np.float32)[np.newaxis]

                ort_inputs_model = {
                    'audio': inputA,
                    'visual': inputV
                }

                start_time = time.time()
                out = model.run(None, ort_inputs_model)[0]
                end_time = time.time()

                # print(f"Asd inference running time: {end_time - start_time:.4f} seconds")
                # print(out)

                ort_inputs_lossAV = {lossAV.get_inputs()[0].name: out}

                start_time = time.time()
                score = lossAV.run(None, ort_inputs_lossAV)
                end_time = time.time()

                # print(f"LossAV inference running time: {end_time - start_time:.4f} seconds")
                # print(score)

                scores.extend(np.array(score).flatten())  # Convert to numpy array before flattening
            
            allScore.append(scores)

        allScore = np.vstack(allScore)  # Stack to make a consistent 2D array
        allScore = np.round(np.mean(allScore, axis=0), 1).astype(float)

        allScores.append(allScore)

    return allScores

# Active Speaker Detection
files = sorted(glob.glob(f"{pycropPath}/*.avi"))
scores = evaluate_network(files)
print(scores)
