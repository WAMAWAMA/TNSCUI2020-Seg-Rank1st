import torch
import segmentation_models_pytorch_4TorchLessThan120 as smp
ecname = 'efficientnet-b5'
cudaa = 1

model =  smp.Unet(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('Unet is ok')

model =  smp.Linknet(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('Linknet is ok')


model =  smp.FPN(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('FPN is ok')

model =  smp.PSPNet(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('PSPNet is ok')

model =  smp.DeepLabV3Plus(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('DeepLabV3Plus is ok')

model =  smp.PAN(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('PAN is ok')

model =  smp.PAN(encoder_name=ecname,
				  encoder_weights='advprop',
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
print('downloading is ok')

