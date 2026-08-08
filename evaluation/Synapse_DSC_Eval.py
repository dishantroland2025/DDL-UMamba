# -*- coding: utf-8 -*-
"""
Synapse_DSC.py

Computes per-organ Dice Similarity Coefficient (DSC) for the Synapse
multi-organ abdominal CT dataset (8 organs), following the same structure
and CLI conventions as the original U-Mamba abdomen evaluation script.

Usage:
    python Synapse_DSC.py --gt_path <labelsTs_dir> --seg_path <predictions_dir> --save_path <output.csv>
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_dice_coefficient

join = os.path.join
basename = os.path.basename
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='')
parser.add_argument('--seg_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(gt_path, x))]
filenames.sort()

# Synapse labels: 1=Aorta, 2=Gallbladder, 3=LKid, 4=RKid, 5=Liver,
# 6=Pancreas, 7=Spleen, 8=Stomach (background = 0)
organ_labels = OrderedDict({
    'Aorta': 1, 'Gallbladder': 2, 'LKid': 3, 'RKid': 4,
    'Liver': 5, 'Pancreas': 6, 'Spleen': 7, 'Stomach': 8
})

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
for organ in organ_labels.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)

    gt_nii = nb.load(join(gt_path, name))
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    for organ, i in organ_labels.items():
        if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
            DSC_i = 1
        elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
            DSC_i = 0
        else:
            organ_i_gt = gt_data == i
            organ_i_seg = seg_data == i
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
        seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_path, index=False)

dsc_cols = [c for c in dataframe.columns if c.endswith('_DSC')]
case_avg_DSC = dataframe[dsc_cols].mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Per-organ mean DSC for {basename(seg_path)}:')
print(case_avg_DSC)
print(f'Overall mean DSC: {case_avg_DSC.mean()}')
print(20 * '<')