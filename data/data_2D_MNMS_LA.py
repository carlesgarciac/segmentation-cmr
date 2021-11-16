import os
import nibabel as nib
import numpy as np

path_data = '/home/carlesgc/Projects/segmentation/MnM-2/training/'
path_out = '/home/carlesgc/Projects/segmentation/train_data/'


for (dirpath, dirnames, filenames) in os.walk(path_data):
    for file in filenames:
        
        if file.split('_')[1] == 'LA':
            dir_patient = file.split('_')[0] 
            if file.split('_')[2].split('.')[0] != 'CINE':
                print(os.path.join(path_data+dir_patient+'/', file))
                if 'gt' not in file:
                    original = nib.load(os.path.join(path_data+dir_patient+'/', file)).get_fdata()
                    affine = nib.load(os.path.join(path_data+dir_patient+'/', file)).affine
                    hdr = nib.load(os.path.join(path_data+dir_patient+'/', file)).header
                    
                    out = nib.Nifti1Image(original, affine, header=hdr)
                    nib.save(out, os.path.join(path_out+'images/', file))
                else:
                    original_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).get_fdata()
                    affine_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).affine
                    hdr_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).header
                    
                    out_gt = nib.Nifti1Image(original_gt, affine_gt, header=hdr_gt)
                    nib.save(out_gt, os.path.join(path_out+'labels/', file.split('.')[0]+'.nii.gz'))