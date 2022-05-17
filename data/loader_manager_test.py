import torchio as tio

import torch

from torch.utils.data import DataLoader

from data.dataset_2D_MNMS_SA_test import get_dataset_SA_test
from data.dataset_2D_MNMS_LA_test import get_dataset_LA_test

from pathlib import Path

import numpy as np


def loader_manager_test(data_view, batch_size, num_workers, create_landmarks=False):

    if data_view == 'SA':
        test_set = Path('/home/carlesgc/Projects/segmentation/test_data_SA/')
        get_dataset = get_dataset_SA_test
        histogram_landmarks_path = '/home/carlesgc/Projects/segmentation/landmarks_SA.npy'
        if create_landmarks == False:
            landmarks = np.load(histogram_landmarks_path)
    elif data_view == 'LA':
        test_set = Path('/home/carlesgc/Projects/segmentation/test_data_LA/')
        get_dataset = get_dataset_LA_test
        histogram_landmarks_path = '/home/carlesgc/Projects/segmentation/landmarks_LA.npy'
        if create_landmarks == False:
            landmarks = np.load(histogram_landmarks_path)

    ## Datasets
    # train_set = Path('/home/carlesgc/Projects/segmentation/train_data_SA/')
    images_dir = test_set / 'images'
    labels_dir = test_set / 'labels'
    image_paths = sorted(images_dir.glob('*.nii.gz'))

    if create_landmarks == True:
        landmarks = tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path,
        )

    testing_transform = tio.Compose([
        tio.CropOrPad((256,256,1)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.HistogramStandardization({'mri': landmarks}),
        tio.RescaleIntensity(out_min_max=(0,1)),
        # tio.OneHot()
    ])
    
    test_subjects = get_dataset(images_dir, labels_dir)
    num_subjects = len(test_subjects)
    

    test_set = tio.SubjectsDataset(
        test_subjects, transform=testing_transform)

    print('Testing set:', len(test_set), 'subjects')

    ## Loaders
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader
