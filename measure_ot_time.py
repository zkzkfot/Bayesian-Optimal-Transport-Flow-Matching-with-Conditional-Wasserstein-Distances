# measure_ot_time.py

import time
import os
import csv
import argparse

import torch
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import ot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
channels = 3
img_size = 32
dim = 3 * 32 * 32  # 3072 (CIFAR10)


def compute_matching_timing(z, gt, observation,
                            ot_mode, sinkhorn_eps, sinkhorn_iter, sinkhorn_trunc_iter):
    """
    학습 코드의 compute_matching에서
    - random: 랜덤 매칭 (OT solver X)
    - emd / sinkhorn_full / sinkhorn_trunc: OT solver 호출
    부분만 떼어 온 버전. 여기서는 한 번 호출에 걸린 시간만 리턴.
    """
    B, dim_ = z.shape
    assert dim_ == dim

    if ot_mode == 'random':
        # OT solver 안 씀 → 시간 0
        perm = torch.randperm(B, device=z.device)
        _ = z - gt[perm]
        return 0.0

    # OT 쓰는 경우: source/target 구성 (학습 코드와 동일)
    source = torch.cat((z, observation), 1)
    target = torch.cat((gt, observation), 1)

    u = ot.unif(B)
    v = ot.unif(B)

    C = torch.cdist(source, target) ** 2
    C_np = C.detach().cpu().numpy()

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

    # g_x까지 계산 (시간에는 거의 영향 없음, 학습 코드와 동일하게 맞추는 용도)
    plan_t = torch.tensor(plan, device=z.device, dtype=torch.float32)
    ind2 = torch.argmax(plan_t, dim=1)
    _ = z - gt[ind2]

    return elapsed


def measure_ot_for_cifar(ot_mode, beta, batchOT,
                         sinkhorn_eps, sinkhorn_iter, sinkhorn_trunc_iter,
                         num_batches, out_dir, tag=None):
    """
    CIFAR10 + 학습 코드와 동일한 OT 세팅으로
    - 각 OT 호출별 걸린 시간 (per-call)
    - 전체 합 / 평균
    을 모두 CSV로 저장한다.
    """
    os.makedirs(out_dir, exist_ok=True)

    # CIFAR10 train set에서 batchOT씩 샘플
    cifar = td.CIFAR10('cifar10', train=True,
                       transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset=cifar, batch_size=batchOT, shuffle=True)

    num_classes = 10

    total_time = 0.0
    n_calls = 0
    per_call_logs = []   # ← 여기 각 호출마다 (call_idx, elapsed) 저장

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= num_batches:
            break

        x = x.to(device)  # (B,3,32,32)
        B = x.size(0)
        if B < batchOT:
            # 마지막 batch가 batchOT보다 작을 수 있으니 스킵
            continue

        gt = x.view(B, -1)  # (B,3072)
        y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float().to(device)
        observation = beta * y_onehot

        z = torch.randn((B, dim), device=device)

        elapsed = compute_matching_timing(
            z=z,
            gt=gt,
            observation=observation,
            ot_mode=ot_mode,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iter=sinkhorn_iter,
            sinkhorn_trunc_iter=sinkhorn_trunc_iter
        )

        total_time += elapsed
        n_calls += 1
        per_call_logs.append((n_calls, float(elapsed)))  # call_idx는 1부터

        print(f"[{ot_mode}] call {n_calls} (batch {b_idx}): OT time = {elapsed:.6f}s")

    if n_calls == 0:
        avg_time = 0.0
    else:
        avg_time = total_time / n_calls

    if tag is None:
        tag = f"{ot_mode}_beta{beta}_bOT{batchOT}"

    # 1) 요약 파일: 전체 / 평균
    summary_path = os.path.join(out_dir, f"ot_timing_{tag}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ot_mode", "beta", "batchOT",
                         "total_ot_time_sec", "num_ot_calls", "avg_ot_time_per_call_sec"])
        writer.writerow([ot_mode, beta, batchOT,
                         total_time, n_calls, avg_time])

    # 2) 디테일 파일: 각 호출별 시간
    detail_path = os.path.join(out_dir, f"ot_timing_{tag}_detail.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["call_idx", "elapsed_sec"])
        writer.writerows(per_call_logs)

    print(f"\n[SUMMARY] mode={ot_mode}, beta={beta}, batchOT={batchOT}, "
          f"calls={n_calls}, total={total_time:.4f}s, avg={avg_time:.6f}s")
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-call detail to {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ot_mode", type=str,
                        choices=["emd", "sinkhorn_full", "sinkhorn_trunc", "random"],
                        required=True)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--batchOT", type=int, default=500)
    parser.add_argument("--sinkhorn_eps", type=float, default=0.001)
    parser.add_argument("--sinkhorn_iter", type=int, default=1000)
    parser.add_argument("--sinkhorn_trunc_iter", type=int, default=20)
    parser.add_argument("--num_batches", type=int, default=100,
                        help="몇 번 OT를 풀어서 시간 측정할지 (학습 전체와 똑같을 필요는 없음)")
    parser.add_argument("--out_dir", type=str, default="ot_timing_measure")
    parser.add_argument("--tag", type=str, default=None,
                        help="결과 파일 이름 구분용 태그 (optional)")
    args = parser.parse_args()

    measure_ot_for_cifar(
        ot_mode=args.ot_mode,
        beta=args.beta,
        batchOT=args.batchOT,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iter=args.sinkhorn_iter,
        sinkhorn_trunc_iter=args.sinkhorn_trunc_iter,
        num_batches=args.num_batches,
        out_dir=args.out_dir,
        tag=args.tag
    )
