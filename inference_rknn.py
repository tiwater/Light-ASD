import os
import numpy as np
import glob
import cv2
from scipy.io import wavfile
import python_speech_features
import tqdm
from rknnlite.api import RKNNLite
import time

pycropPath = "demo/test/pycrop"

def evaluate_network(files):
    # Initialize RKNN model
    lightAsd = RKNNLite()
    lightAsd.load_rknn('./models/lightASD_i8.rknn')
    lightAsd.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)

    lossAV = RKNNLite()
    lossAV.load_rknn('./models/lossAV.rknn')
    lossAV.init_runtime()

    allScores = []
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
            face = face[56:168, 56:168]  # Central crop
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
                inputA = audioFeature[i * duration * 100:(i+1) * duration * 100, :][np.newaxis].astype(np.float32)
                inputV = videoFeature[i * duration * 25:(i+1) * duration * 25, :, :][np.newaxis].astype(np.float32)

                # Note: RKNN 输入只能接收预定义的结构，音频 shape[1] 则只能为 100, 200, 300, 400, 500, 600。
                # 本测试用例中为简单起见，直接抛弃了零散数据。实际生产中应通过补帧或滑动窗口保证输入结构。
                if inputA.shape[1] % 100 != 0:
                    continue

                inputV = inputV.transpose(0, 2, 3, 1)  # from NCHW to NHWC

                start_time = time.time()
                out = lightAsd.inference(inputs=[inputA, inputV])[0]
                end_time = time.time()
                # print(f"Asd inference running time: {end_time - start_time:.4f} seconds")
                # print(out)

                start_time = time.time()
                score = lossAV.inference(inputs=[out])
                end_time = time.time()
                # print(f"LossAV inference running time: {end_time - start_time:.4f} seconds")
                # print(score)

                score_array = np.array(score)
                scores.extend(score_array.flatten())

            allScore.append(scores)

        max_length = max(len(score) for score in allScore)

        # Note: 因为前面抛弃了部分数据，导致此处数据结构可能不完整，用简略方式补齐。如果希望得到准确分值，应考虑严谨的均值计算方法。
        # 如果数据结构规整，则无需做补齐处理，参见 inference_onnx.py 的计算均值方法。
        allScore_padded = [np.pad(score, (0, max_length - len(score)), 'constant') for score in allScore]
        allScore_padded = np.vstack(allScore_padded)
        allScore = np.round(np.mean(allScore_padded, axis=0), 1).astype(float)

        allScores.append(allScore)

    lossAV.release()
    lightAsd.release()
    return allScores

# Active Speaker Detection
files = sorted(glob.glob(f"{pycropPath}/*.avi"))
scores = evaluate_network(files)
print(scores)
