'''
Metrics for model performance analysis
'''

import torch
from torchmetrics import JaccardIndex, ConfusionMatrix, F1Score

class SegmentationMetric():
    def __init__(self, pred, target, indices):
        self.pred = pred 
        self.target = target
        self.indices = indices
        self.num_classes = indices.max() + 1

        
    def jaccard(self):
        jaccard_per_class = JaccardIndex(task="multiclass", num_classes=self.num_classes, average='none') 
        jpc = jaccard_per_class(self.pred, self.target)[self.indices]

        jaccard_weighted = JaccardIndex(task="multiclass", num_classes=self.num_classes, average='weighted') 
        jw = jaccard_weighted(self.pred, self.target)

        return jpc, jw

    def confusion_matrix(self):
        conf_mat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        cm = conf_mat(self.pred, self.target)[self.indices, :][self.indices, :]

        return cm

    def dice(self):
        dice_per_class = F1Score(task='multiclass', num_classes=self.num_classes, average='none')
        dpc = dice_per_class(self.pred, self.target)[self.indices]

        dice_weighted = F1Score(task='multiclass', num_classes=self.num_classes, average='weighted')
        dw = dice_weighted(self.pred, self.target)

        return dpc, dw


if __name__ == '__main__':
    target = torch.randint(0, 2, (10, 25, 25)) # create target tensor: num_masks x h x w
    pred = target.clone().detach() # create pred tensor: num_masks x h x w
    pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15] # modify preds slightly

    indices = torch.tensor([0,1])

    a = indices.max().int() + 1
    
    segmet = SegmentationMetric(pred, target, indices)

    cm = segmet.confusion_matrix()
    print(cm)

