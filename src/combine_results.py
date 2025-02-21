import os 
from itertools import combinations
import random
import json
from tqdm import tqdm

def combine_results(results_dir, 
                    model_list, 
                    prompt_list, 
                    n_combinations, 
                    pred_filename, 
                    output_dir, 
                    sample=10, 
                    max_bigmodel= 3,
                    big_models = ["llama31-70B", "mistral-8x7B-0p1"],
                    gt_file="/home/lisa/repos/semeval/datasets/all-subjects-tib-core-subjects-Article-Book-Conference-Report-Thesis-en-de-dev_sample1000.csv"):
    """
    Find the best combinations of a list of prompts and model combinations. 
    In the output dir, the results are in csv and arrow files,
    containing the number of combinations in the ensemble, e.g: combi_2.csv, combi_2.arrow.

    The best combination and results is finally written again.
    
    """
    pr_aucs = {}
    # first step: get all the actually existing combinations
    existing_combinations = []
    existing_combinations_big = []
    # print(os.path.abspath(os.curdir))
    cwd = os.getcwd()
    # print(f"{cwd=}")
    results_dir = cwd + "/" + results_dir
    # print(f"{results_dir=}")
    output_dir = cwd + "/" + output_dir
    # print(f"{output_dir=}")
    for model in model_list:
        for prompt in prompt_list:
            print("Checking: " +f"{results_dir}/{model}/{prompt}/{pred_filename}")
            if os.path.exists( f"{results_dir}/{model}/{prompt}/{pred_filename}"):
                if model in model in big_models:
                    existing_combinations_big.append((model, prompt))
                else:
                    existing_combinations.append((model, prompt))
                # print("Exists!")
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
        # print(f"{combination=}\n----\n")
        combi_string = " ".join([f"{results_dir}/{model}/{prompt}/{pred_filename}" for model, prompt in combination])
        summarize_command = "python src/summarize_candidates.py " \
                       + combi_string \
                       + " --output_file " + output_dir + "combi_" + str (n_combinations)  + ".csv" \
                       + " --output_file_eval " + output_dir + "combi_" + str(n_combinations)  + ".arrow" 
        # print("Would run this summarize command: " + summarize_command)
        eval_command = "Rscript src/compute_metrics.r "+ "--ground_truth " + gt_file \
                            + " --predictions " + output_dir + "combi_" + str(n_combinations)  + ".arrow" \
                            + " --ordinal_ranking FALSE" \
                            + " --ignore_unmatched_docs FALSE" \
                            + " --out_folder " + output_dir
        # print("Would run this eval command: "+ eval_command)
        run_summarise = os.popen(summarize_command).read()
        # print(f"{run_summarise=}")
        run_eval = os.popen(eval_command).read()
        # print(f"{run_eval=}")
        try:
            print("Would try to open: " + output_dir + "pr_auc.json")
            with open(output_dir + "pr_auc.json", "r") as f:
                pr_auc_dict[combination] = json.load(f)["pr_auc"]
                return pr_auc_dict
        except FileNotFoundError:
            print(f"Error in {combination}, no pr_auc.json found")
        
    for i in tqdm(range(sample)):
        # call run_commands here
        sampled_combinations_small = sorted(random.sample(existing_combinations, n_combinations-max_bigmodel))
        print("Sampled small: ", sampled_combinations_small)
        sampled_combinations_big = sorted(random.sample(existing_combinations_big, max_bigmodel))
        print("Sampled big: ", sampled_combinations_big)
        sampled_combinations = tuple(sampled_combinations_small + sampled_combinations_big)
        # print("Sample: ", i)
        # print("\n".join([str(x) for x in list(enumerate(sampled_combinations))]))
        pr_aucs = run_commands(sampled_combinations, pr_aucs)
    
        
    # get the best three combinations
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
    # print("n:", n)
    return int(factorial(n) / (factorial(k) * factorial(n - k)))

# tmp:
# models = ["llama31-8B", "llama-323B", "mistral-7B", "openhermes-2p5-7B", 
#           "qwen2-7B", "teuken-7B-0p4", 
#           "llama31-70B", "mistral-7B-0p3", "mistral-8x7B-0p1"]
# prompts = ["alltypes-de-abstitle-8-0", 
#           #  "highlemma-fewlabels",
#            "highlemma-manylabels",
#           #  "lowlemma-fewlabels",
#           #  "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", "alltypes-de-abstitle-8-2", "alltypes-de-abstitle-8-3","alltypes-de-abstitle-8-4"]
# o_d = "/assets"
