# -*- coding: utf-8 -*-
"""
Synapse_HD95.py

Computes per-organ 95th-percentile Hausdorff Distance (HD95, in mm) for the
Synapse multi-organ abdominal CT dataset (8 organs), following the same
structure and CLI conventions as the original U-Mamba abdomen evaluation
script. Requires the `surface_distance` package (DeepMind) exposing
`compute_surface_distances` and `compute_robust_hausdorff`.

Usage:
    python Synapse_HD95.py --gt_path <labelsTs_dir> --seg_path <predictions_dir> --save_path <output.csv>
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_robust_hausdorff

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

# HD95 is undefined (infinite) if one of the two masks is empty and the
# other is not; treat both-empty as a perfect match (0 mm) and
# one-empty-one-not as a large penalty, capped at this value (mm), which
# should exceed any real anatomical extent in the volumes evaluated.
HD_PENALTY = 373.13  # ~ diagonal of a 512x512x512 mm volume; adjust if needed

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
for organ in organ_labels.keys():
    seg_metrics['{}_HD95'.format(organ)] = list()

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)

    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()[:3]
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    for organ, i in organ_labels.items():
        gt_empty = np.sum(gt_data == i) == 0
        seg_empty = np.sum(seg_data == i) == 0

        if gt_empty and seg_empty:
            HD95_i = 0.0
        elif gt_empty != seg_empty:
            HD95_i = HD_PENALTY
        else:
            organ_i_gt = gt_data == i
            organ_i_seg = seg_data == i
            surface_distances = compute_surface_distances(
                organ_i_gt, organ_i_seg, case_spacing
            )
            HD95_i = compute_robust_hausdorff(surface_distances, 95)
        seg_metrics['{}_HD95'.format(organ)].append(round(float(HD95_i), 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_path, index=False)

hd_cols = [c for c in dataframe.columns if c.endswith('_HD95')]
case_avg_HD95 = dataframe[hd_cols].mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Per-organ mean HD95 (mm) for {basename(seg_path)}:')
print(case_avg_HD95)
print(f'Overall mean HD95 (mm): {case_avg_HD95.mean()}')
print(20 * '<')