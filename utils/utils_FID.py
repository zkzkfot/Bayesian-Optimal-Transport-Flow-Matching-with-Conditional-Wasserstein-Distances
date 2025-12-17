

# import torch
# import torch.nn.functional as F
# import torchvision.datasets as td
# from torchvision.transforms import transforms
# from torch.utils.data import DataLoader
# import argparse
# from tqdm import tqdm
# import os

# #import utils as ut

# device='cuda'
# dtype=torch.float


# from utils.inception import InceptionV3
# from utils import fid_score as fs

# import numpy as np
# def calc_FID(net,M,z):
    


#     fid_vals = []
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
#     model = InceptionV3([block_idx]).to(device)
#     batch_size=50
#     val = td.CIFAR10('cifar10',train= False,transform=transforms.ToTensor(),download=False)

#     vd = DataLoader(dataset=val,batch_size=M, shuffle=True)
#     data_v= next(iter(vd))
#     gt = data_v[0].to(device)
#     label = data_v[1].to(device)
#     gt = gt.permute(0,2,3,1).cpu().numpy()
#     if gt.shape[-1] == 1:
#         gt = np.concatenate([gt, gt, gt], axis=-1)
#     gt = np.transpose(gt, axes=(0,3,1,2))
#     m1, s1 = fs.calculate_activation_statistics(gt, model, batch_size, 2048, device)

    
#     x= sample(net,M,label,z)
#     gen = torch.clip(x.permute(0, 2, 3, 1), 0, 1).cpu().numpy()
#     if gen.shape[-1] == 1:
#         gen = np.concatenate([gen, gen, gen], axis=-1)
#     gen = np.transpose(gen, axes=(0, 3, 1, 2))
#     m2, s2 = fs.calculate_activation_statistics(gen, model, batch_size, 2048, device)
#     fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)
#     return fid_value
        

    

# def sample(net,M,label,z):
#     with torch.no_grad():
 
 
    
  
  
#         x= z
     
     
#         steps = 100
#         for j in range(steps):
#             t=torch.ones((100,),device=device)*(j/steps)
#             for p in range(M//100):
#                 x[p*100:(p+1)*100]= x[p*100:(p+1)*100]- (1/steps)*net(x[p*100:(p+1)*100],t,label[p*100:(p+1)*100])
        

#         return x[:,:3,:,:].detach()

    
# utils/utils_FID.py

import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

from utils.inception import InceptionV3
from utils import fid_score as fs
import numpy as np


def sample(net, M, label, z, num_steps=100):
    """
    net: 학습된 UNet (flow field)
    M:   샘플 개수 (반드시 100의 배수라고 가정)
    label: (M,) 클래스 라벨
    z:   (M, 3, 32, 32) 초기 노이즈
    num_steps: time discretization step 수
    """
    assert M % 100 == 0, "현재 구현은 M이 100의 배수인 경우만 가정합니다."

    with torch.no_grad():
        x = z.clone()

        steps = num_steps
        for j in range(steps):
            # t \in [0,1], batch 100 기준
            t = torch.ones((100,), device=device) * (j / steps)
            for p in range(M // 100):
                x[p*100:(p+1)*100] = x[p*100:(p+1)*100] - (1.0 / steps) * net(
                    x[p*100:(p+1)*100],
                    t,
                    label[p*100:(p+1)*100]
                )

        return x[:, :3, :, :].detach()


def calc_FID(net, M, z, num_steps=100):
    """
    net: 학습된 UNet (EMA model 등)
    M:   평가에 사용할 샘플 개수 (100의 배수 추천, 예: 10000)
    z:   (M, 3, 32, 32) 고정 noised input
    num_steps: sample()에서 쓸 time step 수
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    batch_size = 50

    # 1) CIFAR10 test set에서 GT 샘플 가져오기
    val = td.CIFAR10('cifar10', train=False,
                     transform=transforms.ToTensor(), download=False)
    vd = DataLoader(dataset=val, batch_size=M, shuffle=True)
    data_v = next(iter(vd))
    gt = data_v[0].to(device)    # (M,3,32,32)
    label = data_v[1].to(device) # (M,)

    # 2) GT 이미지로 mu, sigma 계산
    gt_np = gt.permute(0, 2, 3, 1).cpu().numpy()  # (M,H,W,C)
    if gt_np.shape[-1] == 1:
        gt_np = np.concatenate([gt_np, gt_np, gt_np], axis=-1)
    gt_np = np.transpose(gt_np, axes=(0, 3, 1, 2))  # (M,C,H,W)
    m1, s1 = fs.calculate_activation_statistics(gt_np, model, batch_size, 2048, device)

    # 3) 모델에서 샘플 생성 (num_steps 사용)
    x_gen = sample(net, M, label, z, num_steps=num_steps)
    gen_np = torch.clip(x_gen.permute(0, 2, 3, 1), 0, 1).cpu().numpy()
    if gen_np.shape[-1] == 1:
        gen_np = np.concatenate([gen_np, gen_np, gen_np], axis=-1)
    gen_np = np.transpose(gen_np, axes=(0, 3, 1, 2))
    m2, s2 = fs.calculate_activation_statistics(gen_np, model, batch_size, 2048, device)

    fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


