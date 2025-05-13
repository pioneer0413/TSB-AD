import subprocess
from multiprocessing import Pool
import os, sys
import itertools
import torch
import argparse
from runners import param_sets

SCRIPT_PATH = '/home/hwkang/dev-TSB-AD/TSB-AD/benchmark_exp/Run_Detector_M.py'

def run_script(config):
    args_list, gpu_id = config
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n[GPU {gpu_id}] Args: {args_list}")

    result = subprocess.run(['python', SCRIPT_PATH] + args_list,
                            capture_output=True, text=True, env=env)

    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result.returncode

if __name__=='__main__':

    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='Run scripts with different parameters.')
    parser.add_argument('--param_set_idx', type=str, required=True, help='Path to the file containing parameter sets.')
    parser.add_argument('--run_type', type=int, default=0)
    args = parser.parse_args()

    param_set_idx = args.param_set_idx

    # param_sets_idx에 따라 param_sets 선택
    try:
        param_set = getattr(param_sets, f'param_set_{param_set_idx}')
    except AttributeError:
        print(f"param_set{param_set_idx} is not found in param_sets module.", file=sys.stderr)
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    gpu_ids = range(num_gpus)
    configs = list(zip(param_set, itertools.cycle(gpu_ids)))

    if args.run_type == 0:
        with Pool(processes=8) as pool:
            pool.map(run_script, configs)
    elif args.run_type == 1:
        args_list = param_set[0]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(0)
        print(f"\n[GPU 0] Args: {args_list}")
        result = subprocess.run(['python', SCRIPT_PATH] + args_list,
                                capture_output=True, text=True, env=env)
        print(result.stdout)
        print(result.stderr)
    else:
        print(f"Invalid run_type: {args.run_type}. Choose 0 or 1.", file=sys.stderr)
        sys.exit(1)
    print('All processes completed.')