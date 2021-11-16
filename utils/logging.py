import neptune

import numpy as np

import torch.nn.functional as F



def image_logger_val(data, output, target):
    for image, prediction, gt in zip(data, output, target):
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (1,2,0))
        
        prediction = F.softmax(prediction, dim=0)
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis=0)
        prediction = prediction*1/3
        
        gt = gt.cpu().detach().numpy()
        gt = np.transpose(gt,(1,2,0))
        gt = gt*1/3

        neptune.log_image('validation_predictions', image)
        neptune.log_image('validation_predictions', prediction)
        neptune.log_image('validation_predictions', gt)


def image_logger_train(data, output, target):
    for image, prediction, gt in zip(data, output, target):
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (1,2,0))
        
        prediction = F.softmax(prediction, dim=0)
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis=0)
        prediction = prediction*1/3
        
        gt = gt.cpu().detach().numpy()
        gt = np.transpose(gt,(1,2,0))
        gt = gt*1/3

        neptune.log_image('training_predictions', image)
        neptune.log_image('training_predictions', prediction)
        neptune.log_image('training_predictions', gt)