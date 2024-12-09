# import argparse
# import os

# import numpy as np
# import torch
# import torchvision.transforms as tfs
# import torchvision.utils as vutils
# from PIL import Image
# from tqdm import tqdm

# from metrics import psnr, ssim
# from models.C2PNet import C2PNet

# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset_name', help='name of dataset', choices=['indoor', 'outdoor'],
#                     default='indoor')
# parser.add_argument('--save_dir', type=str, default='dehaze_images', help='dehaze images save path')
# parser.add_argument('--save', action='store_true', help='save dehaze images')
# opt = parser.parse_args()

# dataset = opt.dataset_name

# if opt.save:
#     if not os.path.exists(opt.save_dir):
#         os.mkdir(opt.save_dir)
#     output_dir = os.path.join(opt.save_dir, dataset)
#     print("pred_dir:", output_dir)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)

# if dataset == 'indoor':
#     haze_dir = 'data/SOTS/indoor/hazy/'
import argparse
import os

import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from metrics import psnr, ssim
from models.C2PNet import C2PNet

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', help='name of dataset', choices=['indoor', 'outdoor'],
                    default='indoor')
parser.add_argument('--save_dir', type=str, default='dehaze_images', help='dehaze images save path')
parser.add_argument('--save', action='store_true', help='save dehaze images')
opt = parser.parse_args()

dataset = opt.dataset_name

if opt.save:
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    output_dir = os.path.join(opt.save_dir, dataset)
    print("pred_dir:", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

if dataset == 'indoor':
    haze_dir = 'data/SOTS/indoor/hazy/'
    clear_dir = 'data/SOTS/indoor/clear/'
    model_dir = 'trained_models/ITS.pk'
elif dataset == 'outdoor':
    haze_dir = 'data/SOTS/outdoor/hazy/'
    clear_dir = 'data/SOTS/outdoor/clear/'
    model_dir = 'trained_models/OTS.pkl'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the model
net = C2PNet(gps=2, blocks=10)

# Load the model checkpoint
ckp = torch.load(model_dir)
# Adjust for DataParallel if necessary
if 'module.' in list(ckp['model'].keys())[0]:
    new_state_dict = {}
    for k, v in ckp['model'].items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    net.load_state_dict(new_state_dict)
else:
    net.load_state_dict(ckp['model'])

net = net.to(device)
net.eval()
psnr_list = []
ssim_list = []

# Define image transformation with normalization
transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
])

for im in tqdm(os.listdir(haze_dir)):
    try:
        haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
        if dataset == 'indoor' or dataset == 'outdoor':
            clear_im = im.split('_')[0] + '.png'
        else:
            clear_im = im
        clear = Image.open(os.path.join(clear_dir, clear_im)).convert('RGB')
        
        haze1 = transform(haze)[None, ::].to(device)
        clear_no = transform(clear)[None, ::]
        
        with torch.no_grad():
            pred = net(haze1)
        
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        pp = psnr(pred.cpu(), clear_no)
        ss = ssim(pred.cpu(), clear_no)
        psnr_list.append(pp)
        ssim_list.append(ss)
        
        if opt.save:
            vutils.save_image(ts, os.path.join(output_dir, im))
    
    except Exception as e:
        print(f"Error processing image {im}: {e}")

print(f'Average PSNR is {np.mean(psnr_list)}')
print(f'Average SSIM is {np.mean(ssim_list)}')