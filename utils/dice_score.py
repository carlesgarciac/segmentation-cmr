import torch
from torch import Tensor
from kornia.utils import one_hot
import torch.nn.functional as F

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
    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[:,1:,:,:], target[:,1:,:,:], reduce_batch_first=False)

def dice_loss_LV(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[:,1,:,:], target[:,1,:,:], reduce_batch_first=False)

def dice_loss_MYO(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[:,2,:,:], target[:,2,:,:], reduce_batch_first=False)

def dice_loss_RV(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.type(torch.int64)
    target: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    target = target.squeeze()

    input: torch.Tensor = F.softmax(input, dim=1)
    # print(input.size())
    # print(target.size())

    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input[:,3,:,:], target[:,3,:,:], reduce_batch_first=False)




# def dice_all(prediction, label):
#     label = label.type(torch.int64)
#     # pdb.set_trace()
#     # loss = kornia.losses.dice_loss(prediction, label)

#     loss_dice = (dice_score_lv(prediction,label) + dice_score_myo(prediction,label) + dice_score_rv(prediction,label))/3

#     loss = 1-loss_dice

#     return loss

# def dice_score_lv(prediction, label):
#     label = label.type(torch.int64)
#     eps: float = 1e-8
#     input_soft: torch.Tensor = F.softmax(prediction, dim=1)

#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(
#         label, num_classes=prediction.shape[1],
#         device=prediction.device, dtype=prediction.dtype)

#     input_soft = input_soft[:,1,:,:]
#     target_one_hot = target_one_hot[:,1,:,:]

#     # compute the actual dice score
#     dims = (1, 2)
#     intersection = torch.sum(input_soft * target_one_hot, dims)
#     cardinality = torch.sum(input_soft + target_one_hot, dims)

#     dice_score = 2. * intersection / (cardinality + eps)
#     return torch.mean(dice_score)

# def dice_score_myo(prediction, label):
#     label = label.type(torch.int64)
#     eps: float = 1e-8
#     input_soft: torch.Tensor = F.softmax(prediction, dim=1)

#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(
#         label, num_classes=prediction.shape[1],
#         device=prediction.device, dtype=prediction.dtype)

#     input_soft = input_soft[:,2,:,:]
#     target_one_hot = target_one_hot[:,2,:,:]

#     # compute the actual dice score
#     dims = (1, 2)
#     intersection = torch.sum(input_soft * target_one_hot, dims)
#     cardinality = torch.sum(input_soft + target_one_hot, dims)

#     dice_score = 2. * intersection / (cardinality + eps)
#     return torch.mean(dice_score)

# def dice_score_rv(prediction, label):
#     label = label.type(torch.int64)
#     eps: float = 1e-8
#     input_soft: torch.Tensor = F.softmax(prediction, dim=1)

#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(
#         label, num_classes=prediction.shape[1],
#         device=prediction.device, dtype=prediction.dtype)

#     input_soft = input_soft[:,3,:,:]
#     target_one_hot = target_one_hot[:,3,:,:]

#     # compute the actual dice score
#     dims = (1, 2)
#     intersection = torch.sum(input_soft * target_one_hot, dims)
#     cardinality = torch.sum(input_soft + target_one_hot, dims)

#     dice_score = 2. * intersection / (cardinality + eps)
#     return torch.mean(dice_score)

# def get_dice_score(output, target, epsilon=1e-9):
#     SPATIAL_DIMENSIONS = 2, 3
#     # print("Output", output.shape)
#     # print("Target", target.shape)
#     p0 = output
#     # p0 = output['out'] #Deeplav 
#     g0 = target
#     # print(p0)
#     p1 = 1 - p0
#     g1 = 1 - g0
#     tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
#     fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
#     fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
#     num = 2 * tp
#     denom = 2 * tp + fp + fn + epsilon
#     dice_score = num / denom
#     return dice_score

# def get_dice_loss(output, target):
#     return 1 - get_dice_score(output, target)