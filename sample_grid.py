# sample_grid.py
import argparse
import os
import math

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as td
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image

from utils.unet_oai import UNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
channels = 3
img_size = 32


def get_UNET():
    # cOT_CIFAR.py 에서 쓰던 구조 그대로
    return UNetModel(
        in_channels=channels,
        out_channels=channels,
        num_res_blocks=2,
        num_classes=10,
        image_size=img_size,
        model_channels=256,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=(16,),
    ).to(device)


def clean_state_dict(state):
    """SWA / DataParallel 로 저장된 ckpt를 UNet에 맞게 정리."""
    new_state = {}
    for k, v in state.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    return new_state


@torch.no_grad()
def sample_images(net, num_samples=100, num_steps=100, seed=1234):
    """
    cOT_CIFAR에서 쓰던 sampler를 그대로 구현.
    num_samples : 생성할 이미지 개수 (100의 배수 권장)
    num_steps   : flow 적분 step 수
    """
    assert num_samples % 100 == 0, "num_samples는 100의 배수로 설정하세요."

    torch.manual_seed(seed)

    # CIFAR-10 test set에서 라벨을 가져와서 conditional sampling
    val = td.CIFAR10("cifar10", train=False,
                     transform=transforms.ToTensor(), download=True)
    vd = DataLoader(val, batch_size=num_samples, shuffle=True)
    data_v = next(iter(vd))
    labels = data_v[1].to(device)  # (num_samples,)

    # 초기 z ~ N(0, I)
    x = torch.randn((num_samples, 3, 32, 32), device=device)

    for j in range(num_steps):
        t = torch.ones((num_samples,), device=device) * (j / num_steps)
        # 원래 코드처럼 100개씩 나눠 처리
        for p in range(num_samples // 100):
            beg = p * 100
            end = (p + 1) * 100
            x[beg:end] = x[beg:end] - (1.0 / num_steps) * net(
                x[beg:end],
                t[beg:end],
                labels[beg:end],
            )

    # [0,1] 범위로 클램프
    x = torch.clamp(x, 0.0, 1.0)
    return x


def main(args):
    # 1) 모델 로드
    net = get_UNET()
    raw_state = torch.load(args.model_path, map_location=device)
    state = clean_state_dict(raw_state)
    net.load_state_dict(state, strict=True)
    net.eval()

    # 2) 샘플링
    imgs = sample_images(
        net,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    # 3) 그리드로 저장
    #   nrow은 대충 sqrt(num_samples) 로
    nrow = int(math.sqrt(args.num_samples))
    grid = make_grid(imgs, nrow=nrow, padding=2)  # [0,1] 그대로 저장

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    save_image(grid, args.out_path)

    print(f"Saved sample grid: {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="학습된 모델 .pt 경로 (예: cOT_cifar_emd_beta1/nets1/net_480.pt)")
    parser.add_argument("--out_path", type=str, default="samples_grid.png",
                        help="저장할 이미지 경로")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="생성할 이미지 개수 (100의 배수 추천)")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="flow 적분 step 수 (학습 때랑 비슷하게 100 정도)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="z, 라벨 샘플링용 seed")
    args = parser.parse_args()
    main(args)
