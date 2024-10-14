import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
        self.FC = nn.Linear(128, 1)  # Output a single logit for binary classification

    def forward(self, x, labels=None, r=1):
        x = x.squeeze(1)
        x = self.FC(x)  # This outputs a tensor of shape [batch_size, 1]

        if labels is None:
            predScore = torch.sigmoid(x).view(-1).detach().cpu()  # Get probabilities for inspecting, no need in loss
            return predScore
        else:
            x1 = x / r  # Normalize logits if necessary
            x1 = x1.squeeze()  # Ensure size is [batch_size] for the loss
            nloss = self.criterion(x1, labels.float())  # Match the size with labels
            predScore = torch.sigmoid(x)  # Compute probabilities if needed
            predLabel = torch.round(predScore)
            correctNum = (predLabel.view(-1) == labels.float()).sum().float()
            return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
    def __init__(self):
        super(lossV, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
        self.FC = nn.Linear(128, 1)  # Output a single logit for binary classification

    def forward(self, x, labels, r=1):
        x = x.squeeze(1)
        x = self.FC(x)
        x = x / r  # Normalize logits
        x = x.squeeze()  # Ensure size is [batch_size]
        
        nloss = self.criterion(x, labels.float())  # Match the size with labels
        return nloss