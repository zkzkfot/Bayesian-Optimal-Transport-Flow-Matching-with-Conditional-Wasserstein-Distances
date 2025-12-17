import argparse
import os
import csv
import time  # ⬅ 시간 측정을 위해 추가
import torch
from tqdm import tqdm

from utils.utils_FID import calc_FID  # num_steps 인자 받도록 이미 수정된 버전이라고 가정
from utils.unet_oai import UNetModel

device = 'cuda' if torch.cuda.is_available() else 'device'
channels = 3
img_size = 32


def get_UNET():
    # cOT_CIFAR.py에서 쓰던 설정 그대로
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


# ✅ 예전에 쓰던 clean_state_dict 그대로 이식
def clean_state_dict(state):
    new_state = {}
    for k, v in state.items():
        # 1) AveragedModel의 n_averaged 제거
        if k == "n_averaged":
            continue

        # 2) "module." prefix 제거 (AveragedModel/DP 래핑된 경우)
        if k.startswith("module."):
            new_key = k[7:]
        else:
            new_key = k

        new_state[new_key] = v
    return new_state


def main(args):
    # 1) 모델 로드
    net = get_UNET()

    # 🔥 AveragedModel / DP 세이브된 체크포인트 정리
    raw_state = torch.load(args.model_path, map_location=device)
    clean_state = clean_state_dict(raw_state)

    net.load_state_dict(clean_state, strict=True)
    net.eval()

    # 2) 고정 z 샘플 생성 (모든 num_steps에서 동일 z 사용)
    torch.manual_seed(args.seed)
    M = args.num_samples
    assert M % 100 == 0, "num_samples는 100의 배수로 설정하세요."
    z = torch.randn((M, 3, 32, 32), device=device)

    # 3) num_steps 리스트 파싱
    steps_list = [int(s) for s in args.steps.split(',')]

    # 4) 각 num_steps마다 FID + 시간 측정
    results = []
    for steps in steps_list:
        print(f"\n[Eval] model={args.model_path}, num_steps={steps}")

        start_time = time.time()
        fid = calc_FID(net, M, z.clone(), num_steps=steps)
        elapsed = time.time() - start_time

        print(f"FID (steps={steps}) = {fid:.4f}")
        print(f"Elapsed time for {M} samples @ steps={steps}: {elapsed:.3f} sec")

        # steps, fid, elapsed 모두 저장
        results.append((steps, fid, elapsed))

    # 5) 결과 저장
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag if args.tag is not None else "default"
    out_path = os.path.join(args.out_dir, f"fid_numsteps_{tag}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # ⬅ time_sec 컬럼 추가
        writer.writerow(["model_path", "tag", "num_steps", "FID", "time_sec"])
        for steps, fid, elapsed in results:
            writer.writerow([args.model_path, tag, steps, fid, elapsed])

    print(f"\n[Done] Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="학습된 모델 (.pt) 경로 (EMD / Random / Sinkhorn 전부 가능)")
    parser.add_argument("--num_samples", type=int, default=2000,  # 100의 배수로 두는게 편함
                        help="FID 평가 샘플 수 (100 배수 권장)")
    parser.add_argument("--steps", type=str, default="2,5,20,50,100,200",
                        help="comma-separated num_steps 리스트. 예: '2,5,20,50,100,200'")
    parser.add_argument("--seed", type=int, default=1234,
                        help="z 고정용 seed")
    parser.add_argument("--out_dir", type=str, default="fid_numsteps_results",
                        help="CSV 저장 디렉토리")
    parser.add_argument("--tag", type=str, default=None,
                        help="결과 파일 구분용 태그 (예: test1_emd, test2_sink_bOT200 등)")
    args = parser.parse_args()

    main(args)
