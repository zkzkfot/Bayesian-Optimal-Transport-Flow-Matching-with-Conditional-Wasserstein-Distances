import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from utils.unet_oai import UNetModel
import utils.fid_score as fs
from utils.utils_FID import *
import ot
import os
import time
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

torch.manual_seed(4048)
channels = 3
img_size = 32

def get_UNET():
    # use unet
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
        attention_resolutions=(16,)
    ).to(device)

def compute_matching(z, gt, observation,
                     ot_mode, sinkhorn_eps, sinkhorn_iter, sinkhorn_trunc_iter,
                     ot_stats=None):
    """
    z: (B, dim)
    gt: (B, dim)
    observation: (B, obs_dim)
    ot_stats: dict 같은 곳에 OT 시간 누적하고 싶을 때 넘겨주는 용도
    """
    B, dim = z.shape
    device = z.device

    if ot_mode == 'random':
        # 그냥 independent random matching (OT 안 풂)
        perm = torch.randperm(B, device=device)
        g_x = z - gt[perm]

        if ot_stats is not None:
            ot_stats["ot_calls"] += 1
            # random은 OT 안 풀었으니 ot_time은 0으로 둠
        return g_x

    # OT를 쓰는 경우
    source = torch.cat((z, observation), 1)
    target = torch.cat((gt, observation), 1)

    u = ot.unif(B)
    v = ot.unif(B)

    C = torch.cdist(source, target) ** 2
    # scale down: EMD에는 영향 없고, Sinkhorn에는 eps의 스케일을 정리해 줌
    C = C / (C.mean() + 1e-8)
    C_np = C.detach().cpu().numpy()

    # --- 여기부터 OT solver 시간 측정 ---
    start_ot = time.time()

    if ot_mode == 'emd':
        plan = ot.emd(u, v, C_np)

    elif ot_mode == 'sinkhorn_full':
        plan = ot.sinkhorn(u, v, C_np,
                           reg=sinkhorn_eps,
                           numItermax=sinkhorn_iter)

    elif ot_mode == 'sinkhorn_trunc':
        plan = ot.sinkhorn(u, v, C_np,
                           reg=sinkhorn_eps,
                           numItermax=sinkhorn_trunc_iter)
    else:
        raise ValueError(f"Unknown ot_mode: {ot_mode}")

    elapsed = time.time() - start_ot
    if ot_stats is not None:
        ot_stats["ot_time_sum"] += elapsed
        ot_stats["ot_calls"] += 1
    # -----------------------------------

    plan = torch.tensor(plan, device=device, dtype=torch.float32)

    # ----- 4. 계획(plan)에서 g_x 만드는 방식 분리 -----

    if ot_mode == 'emd':
        # EMD는 거의 permutation matrix이므로 hard matching이 자연스럽다
        ind2 = torch.argmax(plan, dim=1)      # (B,)
        g_x = z - gt[ind2]

    else:
        # Sinkhorn (full / trunc) 에서는 soft barycentric projection 사용
        # row normalization (혹시라도 marginal이 약간 틀어져 있을 때 보호)
        row_sum = plan.sum(dim=1, keepdim=True) + 1e-8
        plan_norm = plan / row_sum           # (B, B)

        # barycentric target: T(z_i) = Σ_j P_ij * x_j
        target_bar = plan_norm @ gt          # (B, dim)

        g_x = z - target_bar

    return g_x

    # ind2 = torch.argmax(plan, dim=1)
    # g_x = z - gt[ind2]
    # return g_x


def main(args):
    start_time = time.time()

    ot_stats = {
                "ot_time_sum": 0.0,
                "ot_calls": 0
            }

    # Load target samples
    cifar = td.CIFAR10('cifar10', transform=transforms.ToTensor(), download=True)
    M = 20000
    N = M
    data = DataLoader(dataset=cifar, batch_size=M, shuffle=False)
    data = next(iter(data))
    dim = 3072
    data_tmp = data[0].view(M, dim).to(device)
    ground_truth = data_tmp.clone()
    label2 = data[1].to(device)
    label = torch.nn.functional.one_hot(data[1].to(device)).float()

    observation = args.beta * label.clone()
    M2 = 2000

    fid_list1 = []

    net = get_UNET()
    optim = torch.optim.Adam(net.parameters(), lr=2e-4)
    averaged_model = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.decay))

    test_z = torch.randn((M2, 3, 32, 32), device=device)
    mse = torch.nn.MSELoss(reduction="sum")
    ntrain = args.n_epochs
    batch = args.batch_data
    batchOT = args.batchOT
    batch_net = args.batch_net
    progress_bar = tqdm(range(ntrain), total=ntrain, position=0, leave=True)

    epoch_logs = []
    for k in progress_bar:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        ind = torch.randperm(M, device=device)
        ind = ind[:batch]
        
        for j in range(batch // batchOT):
            ind_tmp = ind[j * batchOT:(j + 1) * batchOT]
            label_tm = label2[ind_tmp].clone()
            gt = ground_truth[ind_tmp].clone()
            z = torch.randn((batchOT, dim), device=device)
        
            # original code =============================================
            # target = torch.cat((gt, observation[ind_tmp]), 1)
            # source = torch.cat((z, observation[ind_tmp]), 1)
            # t = torch.rand((batchOT,), device=device)
            
            # u, v = ot.unif(source.shape[0]), ot.unif(target.shape[0])
            # C = torch.cdist(source, target) ** 2
            # C = C.cpu().numpy()
            # plan = ot.emd(u, v, C)
            # ind2 = torch.tensor(np.argmax(plan, 1), device=device)
            # g_x = z - gt[ind2]
            
            # #To run random matching replace lines 89-94 with the following:
            # #g_x = z - gt

            #==========================================================

            # 기존에는 여기서 source/target 만들고 emd 호출
            # 이제는 observation도 같이 넘겨서 함수 안에서 처리
            obs_batch = observation[ind_tmp].clone()

            g_x = compute_matching(
                z=z,
                gt=gt,
                observation=obs_batch,
                ot_mode=args.ot_mode,
                sinkhorn_eps=args.sinkhorn_eps,
                sinkhorn_iter=args.sinkhorn_iter,
                sinkhorn_trunc_iter=args.sinkhorn_trunc_iter,
                ot_stats=ot_stats
            )

            t = torch.rand((batchOT,), device=device)
            #==========================================================

            start = z - t.view(-1, 1) * g_x
            
            start = start.detach()
            start = start.reshape(batchOT, channels, img_size, img_size)
            for p in range(batchOT // batch_net):
                optim.zero_grad()
                loss = mse(net(start[p * batch_net:(p + 1) * batch_net], t[p * batch_net:(p + 1) * batch_net],
                                label_tm[p * batch_net:(p + 1) * batch_net]).reshape(batch_net, dim), g_x[p * batch_net:(p + 1) * batch_net])
                l = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optim.step()
                if k > 0:
                    averaged_model.update_parameters(net)
                descr = f"Loss={l:.4f}."
                progress_bar.set_description(descr)

        epoch_time = time.time() - epoch_start
        max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

        epoch_logs.append((k, float(epoch_time), float(max_mem), float(l)))

        # tqdm status에 같이 보여주기
        descr = f"Loss={l:.4f}, epoch_time={epoch_time:.2f}s, max_mem={max_mem:.2f}GB"
        progress_bar.set_description(descr)

        if k % 20 == 0 and k != 0:
            averaged_model.eval()
            fid = calc_FID(averaged_model, M2, test_z.clone())
            print("1", fid)
            fid_list1.append((k, fid))
            averaged_model.train()
        
            torch.save(averaged_model.state_dict(), f"{args.dir_name}/nets{args.beta}/net_{k}.pt")
        
    total_time = time.time() - start_time
    print(f"Total wall-clock time: {total_time/60:.2f} min")

    # ===== epoch_logs 를 CSV로 저장 =====
    # 파일 이름: epoch_logs_{ot_mode}_beta{beta}.csv
    log_filename = f"epoch_logs_{args.ot_mode}_beta{args.beta}.csv"
    log_path = os.path.join(args.dir_name, log_filename)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 헤더
        writer.writerow(["epoch", "epoch_time_sec", "max_mem_GB", "last_batch_loss"])
        # 데이터
        writer.writerows(epoch_logs)

    print(f"Saved epoch logs to {log_path}")
    # ==================================

    # ===== FID 로그를 CSV로 저장 =====
    # fid_{ot_mode}_beta{beta}.csv 형태로 저장
    fid_filename = f"fid_{args.ot_mode}_beta{args.beta}.csv"
    fid_path = os.path.join(args.dir_name, fid_filename)

    with open(fid_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "fid"])
        for epoch, fid_val in fid_list1:
            writer.writerow([int(epoch), float(fid_val)])

    print(f"Saved FID logs to {fid_path}")
    # ==================================

        # ----- OT solver 시간 요약 저장 -----
    avg_ot_time = None
    if ot_stats["ot_calls"] > 0:
        avg_ot_time = ot_stats["ot_time_sum"] / ot_stats["ot_calls"]
    else:
        avg_ot_time = 0.0

    ot_time_filename = f"ot_timing_{args.ot_mode}_beta{args.beta}_bOT{args.batchOT}.csv"
    ot_time_path = os.path.join(args.dir_name, ot_time_filename)

    with open(ot_time_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ot_mode", "beta", "batchOT",
                         "total_ot_time_sec", "num_ot_calls", "avg_ot_time_per_call_sec"])
        writer.writerow([
            args.ot_mode,
            args.beta,
            args.batchOT,
            ot_stats["ot_time_sum"],
            ot_stats["ot_calls"],
            avg_ot_time
        ])

    print(f"[OT TIMING] mode={args.ot_mode}, batchOT={args.batchOT}, "
          f"total={ot_stats['ot_time_sum']:.2f}s, "
          f"calls={ot_stats['ot_calls']}, avg={avg_ot_time:.6f}s")
    # -----------------------------------


    print(fid_list1)
    torch.save(averaged_model.state_dict(), f"{args.dir_name}/nets{args.beta}/net_{k}.pt")
    torch.save(net.state_dict(), f"{args.dir_name}/nets{args.beta}/net{args.beta}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument('--batch_data', type=int, default=20000, help='Dataloader Batch Size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--beta', type=int, default=1, help='Value for beta')
    parser.add_argument('--batchOT', type=int, default=500, help='Batch size for OT computation')
    parser.add_argument('--decay', type=float, default=0.9999, help='Decay value for SWA')
    parser.add_argument('--batch_net', type=int, default=100, help='Batch size for networks')
    parser.add_argument('--dir_name', type=str, default="cOT", help='Directory name for saving files')
    parser.add_argument('--ot_mode', type=str,
                        choices=['emd', 'sinkhorn_full', 'sinkhorn_trunc', 'random'],
                        default='emd',
                        help='Type of OT / matching used')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.01,
                        help='Entropic regularization epsilon for Sinkhorn')
    parser.add_argument('--sinkhorn_iter', type=int, default=200,
                        help='Max iterations for full Sinkhorn')
    parser.add_argument('--sinkhorn_trunc_iter', type=int, default=20,
                        help='Max iterations for truncated Sinkhorn')
    args = parser.parse_args()
    save_dir = os.path.join(args.dir_name, f'nets{args.beta}')
    os.makedirs(save_dir, exist_ok=True)
    main(args)
