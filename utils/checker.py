from torchvision import transforms

import torchio as tio


def dataloader_checker(train_loader):
        for batch_idx, batch in enumerate(train_loader):
            image = batch['mri'][tio.DATA]
            image = image[0,:,:,:,0]
            # print(image.shape)

        # print(image.shape)  
        # img = np.transpose(image, (1,2,0))
        # image = image*255/np.max(image)
        img = transforms.ToPILImage()(image)
        # img = img*255/np.max(img)
        img.save('debug.png')