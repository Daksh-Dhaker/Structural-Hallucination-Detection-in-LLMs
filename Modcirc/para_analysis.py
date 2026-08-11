import numpy as np
import os
import json
import sys
from contextlib import contextmanager

# Create a custom context manager to capture prints
@contextmanager
def capture_prints(output_file):
    class PrintCapture:
        def write(self, text):
            with open(output_file, 'a') as f:
                f.write(text)
            sys.__stdout__.write(text)
        def flush(self):
            sys.__stdout__.flush()

    original_stdout = sys.stdout
    sys.stdout = PrintCapture()
    try:
        yield
    finally:
        sys.stdout = original_stdout

output_file = "parameter_analysis_results.txt"
# Clear the file if it exists
open(output_file, 'w').close()

# Rest of your code wrapped in the context manager
with capture_prints(output_file):
    root = "/home//ModCirc/saved_results"
    et = ["modcirc", "ablation_circ", "ablation_initparti", "ablation_neuroncut"]
    ed = ["medi_attr", "medi_status", "coreference", "pubmed_summ"]
    scores = ["composability", 'consistency', 'reusability']
    files = [f for f in os.listdir(f"{root}/modcirc_param") if "exp_0_results" in f]
    param_res = {}
    for idx, d in enumerate(ed):
        param_res[d] = {}
        for s in scores:
            param_res[d][s] = []
            for k in [5, 10, 15]:
                for xdim in [32, 64, 128]:
                    for i in range(5):
                        path = f"{root}/modcirc_param"
                        files = [f for f in os.listdir(path) if f"exp_{i}_{s}_scores_x={xdim}" in f and f"tk={k}.json" in f]
                        if len(files) == 0:
                            path = f"{root}/modcirc"
                            files = [f for f in os.listdir(path) if f"exp_{i}_{s}_scores_x={xdim}" in f and f"tk={k}.json" in f]
                        cur_res = json.load(open(f"{path}/{files[0]}"))
                        param_res[d][s].append((k, xdim, float(cur_res[idx])))
    print('Parameter analysis of varying k in 5, 10 and 15:')

    for d in ed:
        print('\nDataset:', d)
        for s in scores:
            print('\tCriterion:', s)
            for xdim in [128]:
                mean = []
                std = []
                for k in [5, 10, 15]:
                    mean.append(np.mean([x[-1] for x in param_res[d][s] if x[0] == k and x[1] == xdim]))
                    std.append(np.std([x[-1] for x in param_res[d][s] if x[0] == k and x[1] == xdim]))
                print('\t\tmean ', mean)
                print('\t\tstd ',std)
    print('---------------------------')

    print('parameter analysis of varying x_dim in 32, 64 and 128:')

    for d in ed:
        print('\nDataset:', d)
        for s in scores:
            print('\tCriterion:', s)
            for k in [15]:
                mean = []
                std = []
                for xdim in [32, 64, 128]:
                    mean.append(np.mean([x[-1] for x in param_res[d][s] if x[0] == k and x[1] == xdim]))
                    std.append(np.std([x[-1] for x in param_res[d][s] if x[0] == k and x[1] == xdim]))
                print('\t\tmean ', mean)
                print('\t\tstd ',std)
