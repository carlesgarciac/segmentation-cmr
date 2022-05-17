import random

from tqdm import tqdm

import pandas as pd

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchio as tio

from utils.dice_score import dice_loss, dice_loss_LV, dice_loss_MYO, dice_loss_RV, dice_cross, hausdorff_distance
from utils.logging import image_logger_val, image_logger_train
from utils.checker import dataloader_checker

from data.loader_manager_test import loader_manager_test

from network.network_manager import network_manager

import statistics

import neptune

from scipy.spatial.distance import directed_hausdorff

## Params
seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Available architectures: UNet, Deeplabv3, transUNet, cenet
network_name = 'cenet'
## Available views: SA, LA 
data_view = 'SA'

model = network_manager(network_name, device)

epochs = int(100)
batch_size = int(1)
num_workers = int(8)
learning_rate = float(0.001)
save_checkpoint = bool(True)
training_split_ratio = 0.8

## Datasets
test_loader = loader_manager_test(data_view, batch_size, num_workers)

## Logging (neptune)
# neptune.init(project_qualified_name="carlesgc/Segmentation")
# neptune.create_experiment(network_name+'_'+data_view)

## Optimer, Loss, lr scheduler, etc
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer)
# loss_function = dice_cross

## Training
def prepare_batch(batch, device):
    inputs = batch['mri'][tio.DATA].to(device).squeeze(4)
    targets = batch['heart'][tio.DATA].to(device).squeeze(4)
    disease = batch['disease']
    vendor = batch['vendor']
    patient = batch['patient']
    return inputs, targets, disease, vendor, patient


## Eval
def test(model, device, dataframe, val_loader):
    model.eval()
    
    dice_all = []
    dice_lv = []
    dice_myo = []
    dice_rv = []

    haus_all = []
    haus_lv = []
    haus_myo = []
    haus_rv = []
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                vepoch.set_description('Validation ')

                data, target, disease, vendor, patient = prepare_batch(batch, device)

                if network_name == 'Deeplabv3' or network_name == 'cenet':
                    data = data.expand(-1,3,-1,-1)

                output = model(data)

                if network_name == 'Deeplabv3':
                    output = output['out']


                dice_lv.append(1-dice_loss_LV(output, target).item())
                dice_myo.append(1-dice_loss_MYO(output, target).item())
                dice_rv.append(1-dice_loss_RV(output, target).item())
                dice_all.append((1-dice_loss_LV(output, target).item()+
                    1-dice_loss_MYO(output, target).item() +
                    1-dice_loss_RV(output, target).item())/3)

                haus_lv.append(hausdorff_distance(output, target, channel=1)[0])
                haus_myo.append(hausdorff_distance(output, target, channel=2)[0])
                haus_rv.append(hausdorff_distance(output, target, channel=3)[0])
                haus_all.append((hausdorff_distance(output, target, channel=1)[0] + 
                    hausdorff_distance(output, target, channel=2)[0] + 
                    hausdorff_distance(output, target, channel=3)[0]))

                data_dataframe = {'Subject': str(patient[0]), 'Disease': str(disease[0]), 'Vendor': str(vendor[0]), 
                    'dice_all': (1-dice_loss_LV(output, target).item() + 1-dice_loss_MYO(output, target).item() + 1-dice_loss_RV(output, target).item())/3, 
                    'dice_LV': 1-dice_loss_LV(output, target).item(), 'dice_Myo': 1-dice_loss_MYO(output, target).item(), 'dice_RV': 1-dice_loss_RV(output, target).item(),
                    'Haus_all': (hausdorff_distance(output, target, channel=1)[0] + hausdorff_distance(output, target, channel=2)[0] + hausdorff_distance(output, target, channel=3)[0]), 
                    'Haus_LV': hausdorff_distance(output, target, channel=1)[0], 'Haus_Myo': hausdorff_distance(output, target, channel=2)[0], 'Haus_RV': hausdorff_distance(output, target, channel=3)[0]}
                dataframe = dataframe.append(data_dataframe, ignore_index=True)
                
                vepoch.set_postfix(
                    dice_all=statistics.mean(dice_all), 
                    dice_lv=statistics.mean(dice_lv), 
                    dice_myo=statistics.mean(dice_myo),
                    dice_rv=statistics.mean(dice_rv),
                    )
                                    
    
    dice_all = statistics.mean(dice_all)
    dice_lv = statistics.mean(dice_lv)
    dice_myo = statistics.mean(dice_myo)
    dice_rv = statistics.mean(dice_rv)


    haus_all = statistics.mean(haus_all)
    haus_lv = statistics.mean(haus_lv)
    haus_myo = statistics.mean(haus_myo)
    haus_rv = statistics.mean(haus_rv)

    # print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
    # print('\nValidation set: Average dice score: {:.4f}\n'.format(dice_all))
    return dataframe, dice_all, dice_lv, dice_rv, dice_myo, haus_all, haus_lv, haus_myo, haus_rv

dataframe = pd.read_csv('/home/carlesgc/Projects/segmentation/results.csv')

# Unet LA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_LA/UNet_final.pt'))
# CE-net LA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_LA/cenet_final.pt'))
# # trans Unet LA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_LA/transUNet_final.pt'))
# # Deeplabv3 LA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_LA/Deeplabv3_final.pt'))

# # Unet SA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_SA/UNet_final.pt'))
# # CE-net SA
model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_SA/cenet_final.pt'))
# # trans Unet SA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_SA/transUNet_final.pt'))
# # Deeplabv3 SA
# model.load_state_dict(torch.load('/home/carlesgc/Projects/segmentation/models/MNMS_SA/Deeplabv3_final.pt'))

results, dice_all, dice_lv, dice_rv, dice_myo, haus_all, haus_lv, haus_myo, haus_rv = test(model, device, dataframe, test_loader)
results.to_csv('results_'+network_name+'_'+data_view+'.csv')

print('Dice_all:'+ str(dice_all), 'Dice_LV:'+ str(dice_lv), 'Dice_RV:'+ str(dice_rv), 'Dice_Myo:'+ str(dice_myo), 
    'Haus_all:'+ str(haus_all), 'Haus_LV:'+ str(haus_lv), 'Haus_Myo:'+ str(haus_myo), 'Haus_RV:'+ str(haus_rv))