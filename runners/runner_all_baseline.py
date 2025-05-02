from runners import param_sets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

param_set = param_sets.param_set9
SCRIPT_PATH = '/home/hwkang/TSB-AD/benchmark_exp/Run_Detector_M.py'

def run_script(args_list):
    print(f"\n[Running] Args: {args_list}")
    result = subprocess.run(['python', SCRIPT_PATH] + args_list,
                            capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result.returncode

# Replace sequential execution with parallel execution
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_script, args_list) for args_list in param_set]
    for future in as_completed(futures):
         future.result()

print('All processes completed.')