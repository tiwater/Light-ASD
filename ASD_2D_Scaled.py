import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from loss_2D_Scaled import lossAV, lossV
from model.Model_2D4Scale import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.scaler = GradScaler()   
        self.model = ASD_Model().to(self.device)
        self.lossAV = lossAV().to(self.device)
        self.lossV = lossV().to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)

        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()

            with autocast():  # Use autocast for mixed precision
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device))
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                outsV = self.model.forward_visual_backend(visualEmbed)

                labels = labels[0].reshape((-1)).to(self.device)   # Loss
                nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
                nlossV = self.lossV.forward(outsV, labels, r)
                nloss = nlossAV + 0.5 * nlossV

            # Use the scaler to scale the loss before calling backward
            self.scaler.scale(nloss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            self.optim.zero_grad()  # This was incorrectly self.optim.step() in explanation

            # Update loss statistics
            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            index += len(labels)
            
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush() 

        # After completing an epoch, release cached memory
        torch.cuda.empty_cache()

        sys.stdout.write("\n")      

        return loss/num, lr
  
    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        labels_list = []  # Store labels for comparisons

        with torch.no_grad():
            for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
                with torch.cuda.amp.autocast():  # Use autocast for mixed precision
                    audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device))
                    visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
                    outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                
                    # Get the logits and apply sigmoid for probabilities in mixed precision
                    predScore = torch.sigmoid(outsAV).squeeze(1)
                    predScore = predScore.detach().cpu().numpy()
                    predScores.extend(predScore)
                    labels_list.extend(labels[0].reshape((-1)).cpu().numpy())  # Collect labels

        # Read the evaluation original CSV format lines
        evalLines = open(evalOrig).read().splitlines()[1:]
        scores_series = pandas.Series(predScores)
        labels_series = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
        
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores_series
        evalRes['label'] = labels_series
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        
        cmd = f"python3 -O utils/get_ava_active_speaker_performance.py -g {evalOrig} -p {evalCsvSave}"
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout.split()[2][:5]))

        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=self.device)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)