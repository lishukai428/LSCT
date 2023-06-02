import torch

PATH = '/home/workstation/lishukai/temporal-shift-module/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e25(1)/ckpt.pth.tar'
SD = torch.load(PATH)
sd = SD['state_dict']
count = 0
for k, v in sd.items():
    count+=1
    if count >5:
        break
    print(k, ':', v.size())


