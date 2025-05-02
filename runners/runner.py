import subprocess
from multiprocessing import Pool
import os, sys
import itertools
import torch
import argparse
from runners import param_sets

SCRIPT_PATH = '/home/hwkang/TSB-AD/benchmark_exp/Run_Detector_M.py'

def run_script(config):
    args_list, gpu_id = config
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n[GPU {gpu_id}] Args: {args_list}")

    result = subprocess.run(['python', SCRIPT_PATH] + args_list,
                            capture_output=True, text=True, env=env)

    #print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result.returncode

if __name__=='__main__':

    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='Run scripts with different parameters.')
    parser.add_argument('--param_set_idx', type=int, default=0,
                        help='Path to the file containing parameter sets.')
    args = parser.parse_args()

    param_set_idx = args.param_set_idx

    # param_sets_idx에 따라 param_sets 선택
    try:
        param_set = getattr(param_sets, f'param_set{param_set_idx}')
    except AttributeError:
        print(f"param_set{param_set_idx} is not found in param_sets module.", file=sys.stderr)
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    gpu_ids = range(num_gpus)
    configs = list(zip(param_set, itertools.cycle(gpu_ids)))

    with Pool(processes=8) as pool:
        pool.map(run_script, configs)
    print('All processes completed.')