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
        self.num_classes = indices.max().item() + 1
        
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

    for pred, trgt in zip(pred_list, trgt_list):
       
        seg_met = SegmentationMetric(pred, trgt, indices)
        jaccard_per_class_i, jaccard_weighted_i = seg_met.jaccard()
        
        jaccard_per_class = jaccard_per_class + jaccard_per_class_i
        jaccard_weighted = jaccard_weighted + jaccard_weighted_i

        confusion_matrix = confusion_matrix + seg_met.confusion_matrix()

        dice_per_class_i, dice_weighted_i = seg_met.jaccard()
        dice_per_class = dice_per_class + dice_per_class_i
        dice_weighted = dice_weighted + dice_weighted_i

        pred = convert_binary_safe(pred, map)
        trgt = convert_binary_safe(trgt, map)

        seg_met_bin = SegmentationMetric(pred, trgt, torch.tensor([0,1]))
        jaccard_per_class_bin_i, jaccard_weighted_bin_i = seg_met_bin.jaccard()
        
        jaccard_per_class_bin = jaccard_per_class_bin + jaccard_per_class_bin_i
        jaccard_weighted_bin = jaccard_weighted_bin + jaccard_weighted_bin_i

        confusion_matrix_bin = confusion_matrix_bin + seg_met_bin.confusion_matrix()

        dice_per_class_bin_i, dice_weighted_bin_i = seg_met_bin.jaccard()
        dice_per_class_bin = dice_per_class_bin + dice_per_class_bin_i
        dice_weighted_bin = dice_weighted_bin + dice_weighted_bin_i

    multi_class = (jaccard_per_class, jaccard_weighted, confusion_matrix, dice_per_class, dice_weighted)
    binary_class = (jaccard_per_class_bin, jaccard_weighted_bin, confusion_matrix_bin, dice_per_class_bin, dice_weighted_bin)

    return multi_class, binary_class
    



if __name__ == '__main__':
    
    bin_map = {0: [0, 3, 4, 5, 6, 7, 8, 9], 1:[1, 2, 10, 11, 12, 13, 14, 15]}

    bin_map = {0:[0,1], 1:[2,3,4]}

    l = []
    for i in range(1000):
        target = torch.randint(0, 5, (2200, 1500)) # create target tensor: num_masks x h x w
        l.append(target)

    l2 = l.copy()

    

    indices = torch.tensor([0,1,2,3,4])

    multi_class, binary_class = compute_metrics(l, l2, indices, map=bin_map)

    jaccard_per_class, jaccard_weighted, confusion_matrix, dice_per_class, dice_weighted = multi_class
    print(jaccard_per_class, 'jaccard_per_class')
    print(jaccard_weighted, 'jaccard_weighted')
    print(confusion_matrix, 'confusion_matrix')
    print(dice_per_class, 'dice_per_class')
    print(dice_weighted, 'dice_weighted')








    # # Test semantic segmentation metrics
    # target = torch.randint(0, 2, (10, 25, 25)) # create target tensor: num_masks x h x w
    # pred = target.clone().detach() # create pred tensor: num_masks x h x w
    # pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15] # modify preds slightly

    # indices = torch.tensor([0,1]), torch.tensor([0,2]) + torch.tensor([0,1]), torch.tensor([0,2])
    # print(indices)
    # # a = indices.max().int() + 1
    
    # # segmet = SegmentationMetric(pred, target, indices)

    # # cm = segmet.jaccard()
    # # print(cm)

    # # # # dummy input for binary convert map
    # # # t = torch.randint(0, 5, (2, 5, 5))
    # # # print(t)
    # # # bin_map = {0:[1, 3], 1:[0,2,4]}
    # # # convert_binary_safe(t, bin_map)