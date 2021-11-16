import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np

def get_dataset_SA(image_dir, label_dir):
    image_paths = sorted(image_dir.glob('*.nii.gz'))
    label_paths = sorted(label_dir.glob('*.nii.gz'))
    assert len(image_paths) == len(label_paths)

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            heart=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    # dataset = tio.SubjectsDataset(subjects)
    # print('Dataset size:', len(dataset), 'subjects')
    return subjects