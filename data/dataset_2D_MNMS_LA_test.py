import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
import pandas as pd

import numpy as np

def get_dataset_LA_test(image_dir, label_dir):
    sheet = pd.read_csv('/home/carlesgc/Projects/segmentation/dataset_information.csv', index_col='SUBJECT_CODE', low_memory=False)

    image_paths = sorted(image_dir.glob('*.nii.gz'))
    label_paths = sorted(label_dir.glob('*.nii.gz'))
    assert len(image_paths) == len(label_paths)

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        patient = str(image_path).split('/')[-1].split('_')[0]
        patient = int(patient)
    
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            heart=tio.LabelMap(label_path),
            disease = sheet.loc[patient].loc["DISEASE"],
            vendor  = sheet.loc[patient].loc["VENDOR"],
            patient = str(patient)
        )
        subjects.append(subject)
        
    # dataset = tio.SubjectsDataset(subjects)
    # print('Dataset size:', len(dataset), 'subjects')
    return subjects