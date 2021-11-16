import os
import nibabel as nib
import numpy as np

path_data = '/home/carlesgc/Projects/segmentation/MnM-2/training/'
path_out = '/home/carlesgc/Projects/segmentation/train_data_SA/'


for (dirpath, dirnames, filenames) in os.walk(path_data):
    for file in filenames:
        
        if file.split('_')[1] == 'SA':
            dir_patient = file.split('_')[0] 
            # print(dir_patient)
            if file.split('_')[2].split('.')[0] != 'CINE':
                # print(os.path.join(path_data+dir_patient+'/', file))
                if 'gt' not in file:
                    original = nib.load(os.path.join(path_data+dir_patient+'/', file)).get_fdata()
                    affine = nib.load(os.path.join(path_data+dir_patient+'/', file)).affine
                    hdr = nib.load(os.path.join(path_data+dir_patient+'/', file)).header

                    for i in range(original.shape[2]):
                        out = nib.Nifti1Image(original, affine, header=hdr)
                        nib.save(out, os.path.join(path_out+'images/', file.split('.')[0]+'_'+format(i,"04")+'.nii.gz'))
                else:
                    original_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).get_fdata()
                    affine_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).affine
                    hdr_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).header
                    
                    for i in range(original_gt.shape[2]):
                        out_gt = nib.Nifti1Image(original_gt, affine_gt, header=hdr_gt)
                        nib.save(out_gt, os.path.join(path_out+'labels/', file.split('.')[0]+'_'+format(i,"04")+'.nii.gz'))