from network.unet2D import UNet
from network.deeplabv3 import createDeepLabv3
from network.transUNet import transUNet
from network.cenet import CE_Net_

def network_manager(network_name, device):

    if network_name == 'UNet':
        model = UNet()

    elif network_name == 'Deeplabv3':
        model = createDeepLabv3()

    elif network_name == 'transUNet':
        model = transUNet()

    elif network_name == 'cenet':
        model = CE_Net_()

    model.to(device)
    return model