import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch

class AccuracyCalculator:
    def __init__(self,tolerance=3,discard_token=6,device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tol=tolerance
        self.discard_token=discard_token
        self.device=device
    
    def calculateAccMulti2DFromProbability(self,predProb,targetLabel):
        # Get the predicted classes
        predLabel=predProb.argmax(dim=-1)

        return self.calculateAccMulti2DFromLabel(predLabel,targetLabel)
    
    def calculateAccMulti2DFromLabel(self,predLabel,targetLabel):
        """
        predLabel: tensor of shape (B, N, 2)
        target: tensor of shape (B, N, 2)
        """
        if predLabel.shape[1]>targetLabel.shape[1]:
            predLabel=predLabel[:,:targetLabel.shape[1]]
        if targetLabel.shape[1]>predLabel.shape[1]:
            targetLabel=targetLabel[:,:predLabel.shape[1]]

        mask=(targetLabel>self.discard_token).any(axis=-1)
        mask=mask.to(targetLabel.device)
        mask=mask.unsqueeze(dim=-1).repeat(1,1,2)

        return self.calculateAccMultiFromLabel(predLabel=predLabel,targetLabel=targetLabel,mask=mask)

    def calculateAccMultiFromLabel(self,predLabel,targetLabel,mask=None):
        """
        pred: tensor of shape (B, N, 2)
        target: tensor of shape (B, N)
        """
        
        N_pred=predLabel.shape[1]
        N_gt=targetLabel.shape[1]
        N_min=min(N_pred,N_gt)
        predLabel=predLabel[:,:N_pred]
        targetLabel=targetLabel[:,:N_pred]

        if mask is None:
            mask=targetLabel>self.discard_token
            mask=mask.to(targetLabel.device)

        # Calculate the number of correct predictions (removing the padding)
        correct = (torch.abs(predLabel-targetLabel)<self.tol)*1*mask
        
        correct=correct.sum()

        # Calculate the total number of predictions excluding the discard token
        total = torch.sum(mask)

        # Calculate the accuracy
        accuracy = float(correct) / float(total)

        return accuracy

if __name__=="__main__":
    pass