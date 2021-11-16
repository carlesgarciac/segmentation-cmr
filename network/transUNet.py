from self_attention_cv.transunet import TransUnet

def transUNet():
    model = TransUnet(in_channels=1, img_dim=256, vit_blocks=8, vit_dim_linear_mhsa_block=512, classes=4)
    return model