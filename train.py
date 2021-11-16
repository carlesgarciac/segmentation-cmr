import random

from tqdm import tqdm

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchio as tio

from utils.dice_score import dice_loss, dice_loss_LV, dice_loss_MYO, dice_loss_RV, dice_cross
from utils.logging import image_logger_val, image_logger_train
from utils.checker import dataloader_checker

from data.loader_manager import loader_manager

from network.network_manager import network_manager

import statistics

import neptune

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
batch_size = int(16)
num_workers = int(8)
learning_rate = float(0.001)
save_checkpoint = bool(True)
training_split_ratio = 0.8

## Datasets
train_loader, val_loader = loader_manager(data_view, training_split_ratio, batch_size, num_workers)

## Logging (neptune)
neptune.init(project_qualified_name="carlesgc/Segmentation")
neptune.create_experiment(network_name+'_'+data_view)

## Optimer, Loss, lr scheduler, etc
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer)
loss_function = dice_cross

## Training
def prepare_batch(batch, device):
    inputs = batch['mri'][tio.DATA].to(device).squeeze(4)
    targets = batch['heart'][tio.DATA].to(device).squeeze(4)
    return inputs, targets

def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch): #enumerate(tqdm(train_loader, desc='Epoch '+str(epoch)+'/'+str(epochs))):
            tepoch.set_description('Epoch '+str(epoch)+'/'+str(epochs))

            data, target = prepare_batch(batch, device)
            optimizer.zero_grad()

            if network_name == 'Deeplabv3' or network_name == 'cenet':
                data = data.expand(-1,3,-1,-1)

            output = model(data)

            if network_name == 'Deeplabv3':
                output = output['out']

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 44 == 0:
                image_logger_train(data, output, target)
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            
            tepoch.set_postfix(loss=loss.mean().item())  #, accuracy=100. * accuracy)
            
    return loss.mean()

## Eval
def eval(model, device, loss_function, val_loader, epoch):
    model.eval()
    val_loss = []
    dice_all = []
    dice_lv = []
    dice_myo = []
    dice_rv = []
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                vepoch.set_description('Validation '+str(epoch)+'/'+str(epochs))

                data, target = prepare_batch(batch, device)

                if network_name == 'Deeplabv3' or network_name == 'cenet':
                    data = data.expand(-1,3,-1,-1)

                output = model(data)

                if network_name == 'Deeplabv3':
                    output = output['out']

                val_loss.append(loss_function(output, target).item()) 
                dice_all.append(1-dice_loss(output, target).item())
                dice_lv.append(1-dice_loss_LV(output, target).item())
                dice_myo.append(1-dice_loss_MYO(output, target).item())
                dice_rv.append(1-dice_loss_RV(output, target).item())

                if batch_idx % 40 == 0:
                    image_logger_val(data, output, target)
                
                vepoch.set_postfix(
                    dice_all=statistics.mean(dice_all), 
                    dice_lv=statistics.mean(dice_lv), 
                    dice_myo=statistics.mean(dice_myo),
                    dice_rv=statistics.mean(dice_rv),
                    loss=statistics.mean(val_loss)
                    )
                                    
    val_loss = statistics.mean(val_loss)
    dice_all = statistics.mean(dice_all)
    dice_lv = statistics.mean(dice_lv)
    dice_myo = statistics.mean(dice_myo)
    dice_rv = statistics.mean(dice_rv)

    # print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
    # print('\nValidation set: Average dice score: {:.4f}\n'.format(dice_all))
    return val_loss, dice_all, dice_lv, dice_rv, dice_myo

for epoch in range(1, epochs):
    train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
    val_loss, val_acc, dice_lv, dice_rv, dice_myo = eval(model, device, loss_function, val_loader, epoch)
    scheduler.step(val_loss)

    if epoch %  5 == 0  and save_checkpoint:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,    
        }, '/home/carlesgc/Projects/segmentation/models/'+'MNMS_'+data_view+'/'+network_name+'_'+format(epoch,"04")+'.pt')

    # dataloader_checker(train_loader)

    neptune.log_metric('train_loss', train_loss)
    neptune.log_metric('val_loss', val_loss)
    neptune.log_metric('val_acc', val_acc)
    neptune.log_metric('dice_lv', dice_lv)
    neptune.log_metric('dice_myo', dice_myo)
    neptune.log_metric('dice_rv', dice_rv)

torch.save(model.state_dict(), '/home/carlesgc/Projects/segmentation/models/'+'MNMS_'+data_view+'/'+network_name+'_final.pt')