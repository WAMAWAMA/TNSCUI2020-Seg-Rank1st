import torch
import torch.nn as nn
import torch.nn.functional as F

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(SR.size(0))

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def get_IOU(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2
    FN = ((SR == 0) + (GT == 1)) == 2

    IOU = float(torch.sum(TP))/(float(torch.sum(TP+FP+FN)) + 1e-6)

    return IOU


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        # m1 = probs.view(num, -1)
        # m2 = targets.view(num, -1)
        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


