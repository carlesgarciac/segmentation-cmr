import torch
from torch import Tensor
from kornia.utils import one_hot
import torch.nn.functional as F
from kornia.losses import hausdorff
from scipy.spatial.distance import directed_hausdorff

## Losses
def dice_cross(prediction, label):
    label = label.type(torch.int64)
    loss_cross = F.cross_entropy(F.softmax(prediction, dim=1).squeeze(), label.cuda().squeeze())
    loss = loss_cross + dice_loss(prediction, label)

    return loss

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def hausdorff_distance(input: Tensor, target: Tensor, channel):
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    input = input.squeeze()

    assert input.size() == target.size()

    hdloss = directed_hausdorff

    
    haus_dist = hdloss(input[channel,:,:].cpu(), target[channel,:,:].cpu())
    
    return haus_dist


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = True):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    input = input.squeeze()
    
    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[1:,:,:], target[1:,:,:], reduce_batch_first=False)

def dice_loss_LV(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    input = input.squeeze()

    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[1,:,:], target[1,:,:], reduce_batch_first=False)

def dice_loss_MYO(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    input = input.squeeze()

    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[2,:,:], target[2,:,:], reduce_batch_first=False)

def dice_loss_RV(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    input = input.squeeze()

    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[3,:,:], target[3,:,:], reduce_batch_first=False)
