# IDS705_Final_Team2

## Purpose:

This repo contains models and analysis tools for semantic segmentation of aerial drone images. 

The ethics and safety of UAVs have been called into question for some time, but with so much environmental and social upside, developing safe flying and landing is of the utmost importance. In order to enhance drone safety, our research is concerned with applying state of the art deep learning models to solve semantic segmentation of drone cameras. To evaluate the performance of our models we used several metrics common in semantic segmentation problems including the jaccard index and dice score and confusion matrices. This repo contains the models as well as tools developed for easy analysis of the models performance. 

<img width="1019" alt="Project Flow Chart" src="proj_flowchart.png">


## Scripts and uses:

The preprocess.py script contains a torch dataset subclass that performs the appropriate data augmentations. takes in 2 inputs, image directory and annotation directory.


Zhanyi add your descrptions of ur files (run models to get outputs which are lists of tensors and save them as .pt files)


The metrics.py script computes all the metrics using the predictions and ground truth. This requires two .pt files which are lists of mask tensors.
