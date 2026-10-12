import numpy as np
import torch
from torch.nn import functional as F


def segmentation_loss(logits,target,positive_weight=4.0):
    valid = target!=255
    if not valid.any():
        raise ValueError('Batch contains no supervised pixels')
    weights = logits.new_tensor([1.0,positive_weight])
    ce = F.cross_entropy(logits,target,weight=weights,ignore_index=255)
    probability = logits.softmax(dim=1)[:,1]
    truth = (target==1).to(probability.dtype)
    probability = probability*valid
    truth = truth*valid
    intersection = (probability*truth).sum(dim=(1,2))
    denominator = probability.sum(dim=(1,2))+truth.sum(dim=(1,2))
    dice_loss = 1-((2*intersection+1)/(denominator+1)).mean()
    return ce+dice_loss


def confusion(prediction,target):
    valid = target!=255
    truth = target==1
    prediction = np.asarray(prediction,dtype=bool)
    return np.asarray([np.sum(prediction&truth&valid),np.sum(prediction&~truth&valid),
                       np.sum(~prediction&truth&valid),np.sum(~prediction&~truth&valid)],dtype=np.int64)


def scores(counts):
    tp,fp,fn,tn = [int(x) for x in counts]
    divide = lambda a,b: float(a/b) if b else None
    return {'iou':divide(tp,tp+fp+fn),'dice':divide(2*tp,2*tp+fp+fn),
            'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'evaluated_pixels':tp+fp+fn+tn}
