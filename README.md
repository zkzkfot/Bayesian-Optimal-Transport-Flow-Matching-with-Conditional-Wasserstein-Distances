# Bayesian-Optimal-Transport-Flow-Matching-with-Conditional-Wasserstein-Distances
This code is based on the paper **Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching** available at https://arxiv.org/abs/2403.18705 [1] and https://github.com/JChemseddine/Conditional_Wasserstein_Distances.

It contains implementation for the experiments of conditional generation task on CIFAR-10 with different OT solvers such as EMD and Sinkhorn.

For question, contact
zkzkfot@unist.ac.kr or .

## REQUIREMENTS
To run the code, you need to make conda environment with :

```conda env create -f environment.yml```

If this yml file does not work, you can see the requirement.txt file which is original code's setting. In our case, we did experiments with RTX4090.

## Quick Start

- Train EMD

```python cOT_CIFAR.py --ot_mode emd --n_epochs 500 --beta 1 --batchOT 500 --dir_name cOT_cifar_emd_bOT500```
- Train Random

```python cOT_CIFAR.py --ot_mode random --n_epochs 500 --beta 1 --batchOT 500 --dir_name cOT_cifar_random_beta1_bOT500```
- Train Sinkhorn

```python cOT_CIFAR.py --ot_mode sinkhorn_full --sinkhorn_eps 0.001 --sinkhorn_iter 1000 --n_epochs 500 --beta 1 --batchOT 100 --dir_name cOT_cifar_sink_eps0_001_bOT100```

- Evaluate EMD with differenc sampling steps

```python eval_num_steps.py --model_path cOT_cifar_emd_bOT500/nets1/net_499.pt --num_samples 2000 --steps "2,5,20,50,100,200" --out_dir fid_numsteps_test --tag test_emd```
- Evaluate Random with differenc sampling steps

```python eval_num_steps.py --model_path cOT_cifar_random_beta1_bOT500/nets1/net_499.pt --num_samples 2000 --steps "2,5,20,50,100,200" --out_dir fid_numsteps_test --tag test_emd```
- Evaluate Sinkhorn with differenc sampling steps

```python eval_num_steps.py --model_path cOT_cifar_sink_eps0_001_bOT100/nets1/net_499.pt --num_samples 2000 --steps "2,5,20,50,100,200" --out_dir fid_numsteps_test --tag test_emd```

- Simulate and calulate OT solving time

```python measure_ot_time.py --ot_mode emd --beta 1 --batchOT 500 --num_batches 100 --out_dir ot_timing_test --tag emd_bOT500```

## References
[1] J. Chemseddine, P. Hagemann, C. Wald, G. Steidl.
Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.

## CITATION
```python
@article{CHWS2024,
      title={Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching}, 
      author={Jannis Chemseddine and Paul Hagemann and Christian Wald and Gabriele Steidl},
      journal={arXiv preprint arXiv:2403.18705}
}
```
