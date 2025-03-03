"""
File name: combine_results.py
Description: Contains code to optimize the selection of modelxprompt combinations.
"""

import os 
from itertools import combinations
import random
import json
from tqdm import tqdm
from argparse import ArgumentParser

def combine_results(results_dir: str, 
                    model_list: list, 
                    prompt_list: list, 
                    n_combinations: int, 
                    pred_filename: str, 
                    output_dir: str, 
                    gt_file: str,
                    sample: int = 10, 
                    max_bigmodel: int = 3,
                    big_models: list = [
                        "llama31-70B", 
                        "mistral-8x7B-0p1"
                        ]
                    ):
    """
    Find the best combinations of a list of prompts and model combinations. 
    In the output dir, the results are in csv and arrow files,
    containing the number of combinations in the ensemble, e.g: combi_2.csv, combi_2.arrow.

    The best combination and results is finally written again.

    Args:
        results_dir (str): The directory to read results from.
        model_list (list): List of models to combine, must correspond to 
                            subfolders in the results directory.
        prompt_list (list): List of prompts to combine, must correspond to 
                            sub-subfolders in the results directory.
        n_combinations (int): How many prompt-model combinations to combine.
        pred_filename (str): The name of the individual prediction files.
        output_dir (str): The directory to write the best combinations to.
        gt_file (str): The ground-truth file, created with preprocess.py.
        sample (int): How many samples to take to find the best combination.
        max_bigmodel (int): How many combinations with a big model to include
                            (Reason: big models are especially heavy in inference time).
        big_models (list): List of models that are considered big models.
    Writes:
        best_combinations_{n_combinations}.json: JSON dump of a list of the best modelxprompt combinations.
        combi_{n_combinations}.csv: The combined results of the best combination.
        The metrics/output files from the eval-pipeline

    """
    pr_aucs = {}
    # first step: get all the actually existing combinations
    existing_combinations = []
    existing_combinations_big = []
    # print(os.path.abspath(os.curdir))
    cwd = os.getcwd()
    results_dir = cwd + "/" + results_dir
    output_dir = cwd + "/" + output_dir
    for model in model_list:
        for prompt in prompt_list:
            print("Checking: " +f"{results_dir}/{model}/{prompt}/{pred_filename}")
            if os.path.exists( f"{results_dir}/{model}/{prompt}/{pred_filename}"):
                if model in model in big_models:
                    existing_combinations_big.append((model, prompt))
                else:
                    existing_combinations.append((model, prompt))
            else:
                print("Does not exist: ", f"{results_dir}/{model}/{prompt}/{pred_filename}")	
    print("Existing combinations: ", len(existing_combinations))
    
    if n_combinations - max_bigmodel > len(existing_combinations):
        print(f"Sample size is too large, using all combinations instead")
        n_combinations = len(existing_combinations) + max_bigmodel
    
    if max_bigmodel > len(existing_combinations_big):
        print(f"Sample size is too large, using all big model combinations instead")
        max_bigmodel = len(existing_combinations_big)
    if max_bigmodel > n_combinations:
        max_bigmodel = n_combinations
        print("Setting max_bigmodel to n_combinations as a too big value was entered for it.")

    # helper function to reduce code duplication
    def run_commands(combination, pr_auc_dict):
        combi_string = " ".join([f"{results_dir}/{model}/{prompt}/{pred_filename}" for model, prompt in combination])
        summarize_command = "python src/summarize_candidates.py " \
                       + combi_string \
                       + " --output_file " + output_dir + "combi_" + str (n_combinations)  + ".csv" 
        eval_command = "Rscript src/R/eval/compute_metrics.r "+ "--ground_truth " + gt_file \
                            + " --predictions " + output_dir + "combi_" + str(n_combinations)  + ".csv" \
                            + " --ordinal_ranking FALSE" \
                            + " --ignore_unmatched_docs FALSE" \
                            + " --out_folder " + output_dir
        run_summarise = os.popen(summarize_command).read()
        run_eval = os.popen(eval_command).read()
        try:
            print("Would try to open: " + output_dir + "pr_auc.json")
            with open(output_dir + "pr_auc.json", "r") as f:
                pr_auc_dict[combination] = json.load(f)["pr_auc"]
                return pr_auc_dict
        except FileNotFoundError:
            print(f"Error in {combination}, no pr_auc.json found")
        
    for i in tqdm(range(sample)):
        sampled_combinations_small = sorted(random.sample(existing_combinations, n_combinations-max_bigmodel))
        print("Sampled small: ", sampled_combinations_small)
        sampled_combinations_big = sorted(random.sample(existing_combinations_big, max_bigmodel))
        print("Sampled big: ", sampled_combinations_big)
        sampled_combinations = tuple(sampled_combinations_small + sampled_combinations_big)
        pr_aucs = run_commands(sampled_combinations, pr_aucs)
    
        
    # get the best ten combinations
    best_combinations = sorted(pr_aucs.items(), key=lambda x: x[1], reverse=True)[:10] 
    print(f"Best combinations: {best_combinations}")
    with open(output_dir + "best_combinations_{}.json".format(n_combinations), "w") as f:
        json.dump(best_combinations, f)
    best_combi = best_combinations[0][0]  
    # write the best combi    
    pr_aucs = run_commands(best_combi, pr_aucs)
    print(f"Best combinations: {best_combinations}")
    return pr_aucs

# Functions helping to determine, how many different combinations of promptsxmodels are possible, when assuming different numbers of combinations
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
    
def possible_combinations(models, prompts, k):
    n=len(models)*len(prompts)
    return int(factorial(n) / (factorial(k) * factorial(n - k)))

def execute():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", help="Directory with the results", type=str, required=True)
    parser.add_argument("--model_list", help="List of model names (short names) to consider. Enter as separate strings", type=str, nargs='+', required=True)
    parser.add_argument("--prompt_list", help="List of prompts (short names) to consider. Enter as separate strings", type=str, nargs='+', required=True)
    parser.add_argument("--n_combinations", help="Number of modelxprompt combinations in the ensemble you want to optimize", type=int, required=True)
    parser.add_argument("--pred_filename", help="Name of the individual prediction files", type=str, required=True)
    parser.add_argument("--output_dir", help="Directory to write the best combinations to.", type=str, required=True)
    parser.add_argument("--gt_file", help="Ground-truth file, created with preprocess.py", type=str, required=True)
    parser.add_argument("--sample", help="How many samples to take to find the best combination (default:10)", type=int, default=10, required=False)
    parser.add_argument("--max_bigmodel", help="How many combinations with a big model to include (default:3)", type=int, default=3, required=False)
    parser.add_argument("--big_models", help="List of models that are considered big models (default: llama31-70B, mistral-8x7B-0p1). Enter as separate strings", type=str, nargs='+', default=["llama31-70B", "mistral-8x7B-0p1"], required=False)

    args = parser.parse_args()

    combine_results(args.results_dir,
                    args.model_list,
                    args.prompt_list,
                    args.n_combinations,
                    args.pred_filename,
                    args.output_dir,
                    args.gt_file,
                    args.sample,
                    args.max_bigmodel,
                    args.big_models)

if __name__=="__main__":
    execute()