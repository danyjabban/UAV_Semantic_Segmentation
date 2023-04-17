'''
Metrics for model performance analysis
'''

import torch
from torchmetrics import JaccardIndex, ConfusionMatrix, F1Score
import numpy as np
import pandas as pd

class SegmentationMetric():
    def __init__(self, pred, target, indices):
        self.pred = pred 
        self.target = target
        self.indices = indices
        self.num_classes = indices.max().item() + 1
        
    def jaccard(self):
        jaccard_per_class = JaccardIndex(task="multiclass", num_classes=self.num_classes, average='none') 
        jpc = jaccard_per_class(self.pred, self.target)#[self.indices]

        jaccard_weighted = JaccardIndex(task="multiclass", num_classes=self.num_classes, average='macro') 
        jw = jaccard_weighted(self.pred, self.target)

        return jpc, jw

    def confusion_matrix(self):
        conf_mat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        cm = conf_mat(self.pred, self.target)#[self.indices, :][self.indices, :]

        return cm

    def dice(self):
        dice_per_class = F1Score(task='multiclass', num_classes=self.num_classes, average='none')
        dpc = dice_per_class(self.pred, self.target)[self.indices]

        dice_weighted = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        dw = dice_weighted(self.pred, self.target)

        return dpc, dw
    
    def check_label_present(self):
        check = torch.zeros(self.num_classes)
        for i in self.indices:
            if i in self.target:
                check[i] = 1
        return check


    

def convert_binary_safe(t, map):
    if (0 in map[1]) and (1 in map[0]):
        t = torch.where(t==0, -1, t)
        t = torch.where(t==1, 0, t)
        t = torch.where(t==-1, 1, t)
    elif 1 in map[0]:
        t = torch.where(t==1, 0, False)
    elif 0 in map[1]:
        t = torch.where(t==0, 1, False)
    for i in map[0]: 
        if i not in [0,1]:
            t = torch.where(t==i, 0, t)
    t = torch.where(t != 0, 1, t)
    return t



def compute_metrics(pred_list, trgt_list, indices, map):
    jaccard_per_class = 0
    jaccard_weighted = 0
    confusion_matrix = 0
    dice_per_class = 0
    dice_weighted = 0

    jaccard_per_class_bin = 0
    jaccard_weighted_bin = 0
    confusion_matrix_bin = 0
    dice_per_class_bin = 0
    dice_weighted_bin = 0

    check_label = torch.zeros(indices.shape[0])

    for pred, trgt in zip(pred_list, trgt_list):
       
        seg_met = SegmentationMetric(pred, trgt, indices)
        jaccard_per_class_i, jaccard_weighted_i = seg_met.jaccard()
        
        jaccard_per_class = jaccard_per_class + jaccard_per_class_i
        jaccard_weighted = jaccard_weighted + jaccard_weighted_i

        confusion_matrix = confusion_matrix + seg_met.confusion_matrix()

        dice_per_class_i, dice_weighted_i = seg_met.dice()
        dice_per_class = dice_per_class + dice_per_class_i
        dice_weighted = dice_weighted + dice_weighted_i

        check_label = check_label + seg_met.check_label_present()

        pred = convert_binary_safe(pred, map)
        trgt = convert_binary_safe(trgt, map)

        seg_met_bin = SegmentationMetric(pred, trgt, torch.tensor([0,1]))
        jaccard_per_class_bin_i, jaccard_weighted_bin_i = seg_met_bin.jaccard()
        
        jaccard_per_class_bin = jaccard_per_class_bin + jaccard_per_class_bin_i
        jaccard_weighted_bin = jaccard_weighted_bin + jaccard_weighted_bin_i

        confusion_matrix_bin = confusion_matrix_bin + seg_met_bin.confusion_matrix()

        dice_per_class_bin_i, dice_weighted_bin_i = seg_met_bin.dice()
        dice_per_class_bin = dice_per_class_bin + dice_per_class_bin_i
        dice_weighted_bin = dice_weighted_bin + dice_weighted_bin_i

    multi_class = (torch.div(jaccard_per_class, check_label), jaccard_weighted/len(pred_list), 
                   confusion_matrix, torch.div(dice_per_class, check_label), dice_weighted/len(pred_list))
    binary_class = (jaccard_per_class_bin/len(pred_list), jaccard_weighted_bin/len(pred_list), 
                    confusion_matrix_bin, dice_per_class_bin/len(pred_list), dice_weighted_bin/len(pred_list))

    return multi_class, binary_class
    



if __name__ == '__main__':
    
    bin_map = {0: [0, 3, 4, 5, 6, 7, 8, 9], 1:[1, 2, 10, 11, 12, 13, 14, 15]}


    pred_list = torch.load('masked_list_resnet18.pt', map_location=torch.device('cpu'))

    target_list = torch.load('ground_truth_list_resnet18.pt')

    indices = torch.arange(17)

    multi_class, binary_class = compute_metrics(pred_list, target_list, indices, map=bin_map)

    jaccard_per_class, jaccard_weighted, confusion_matrix, dice_per_class, dice_weighted = multi_class
    print(jaccard_per_class, 'jaccard_per_class')
    print(jaccard_weighted, 'jaccard_weighted')
    print(confusion_matrix, 'confusion_matrix')
    print(dice_per_class, 'dice_per_class')
    print(dice_weighted, 'dice_weighted')
    confusion_matrix = confusion_matrix.numpy()
    confusion_matrix = pd.DataFrame(confusion_matrix)
    confusion_matrix.to_csv('resnet_cm_multi.csv')

    jaccard_per_class, jaccard_weighted, confusion_matrix, dice_per_class, dice_weighted = binary_class
    print(jaccard_per_class, 'jaccard_per_class')
    print(jaccard_weighted, 'jaccard_weighted')
    print(confusion_matrix, 'confusion_matrix')
    print(dice_per_class, 'dice_per_class')
    print(dice_weighted, 'dice_weighted')





