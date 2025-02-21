from src.combine_results import combine_results
import os

# models = ["llama31-8B", 
#           "llama-32-3B", 
#           "mistral-7B", 
#           "mistral-7B-0p3",
#           "openhermes-2p5-7B", 
#           "qwen2-7B", 
#           "teuken-7B-0p4", 
#         #   "llama31-70B", 
#         #   "mistral-8x7B-0p1"
#           ]
# prompts = ["alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"]

# for i in range(10, 105, 10):
#     out_dir = "results/best_combinations_nobigm/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=100
# )
# models = [
#           "llama31-8B", 
#           "llama-32-3B", 
#           "mistral-7B", 
#           "mistral-7B-0p3",
#           "openhermes-2p5-7B", 
#           # "qwen2-7B", 
#           # "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]

# for i in range(10, 90,10):
#     out_dir = "result_combinations/all_wo_qwen_teuken/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=10
# )

# models = [
#           # "llama31-8B", 
#           # "llama-32-3B", 
#           # "mistral-7B", 
#           # "mistral-7B-0p3",
#           # "openhermes-2p5-7B", 
#           # "qwen2-7B", 
#           # "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#           #  "alltypes-de-abstitle-8-0", 
#           #  "highlemma-fewlabels",
#           #  "highlemma-manylabels",
#           #  "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#           #  "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#           #  "english-0-8",
#           #  "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#           #  "mixed-1-8",
#           #  "mixed-2-12"
#            ]

# for i in range(9, 10):
#     out_dir = "result_combinations/bigmodels/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=50
# )

# models = [
#           # "llama31-8B", 
#           # "llama-32-3B", 
#           # "mistral-7B", 
#           "mistral-7B-0p3",
#           # "openhermes-2p5-7B", 
#           # "qwen2-7B", 
#           # "teuken-7B-0p4", 
#         #   "llama31-70B", 
#         #   "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]

# for i in range(15,16):
#     out_dir = "result_combinations/mistral_0p3/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=1
# )
# for i in range(5, 11, 5):
#     out_dir = "result_combinations/mistral_0p3/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=20
# )
    
# models = [
#           "llama31-8B", 
#           "llama-32-3B", 
#           "mistral-7B", 
#           "mistral-7B-0p3",
#           "openhermes-2p5-7B", 
#           "qwen2-7B", 
#           "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#          #   "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#          #   "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]

# for i in range(15,16):
#     out_dir = "result_combinations/mistral7x8/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=1
# )

# for m in models:
#    for i in range(15,16):
#       out_dir = "result_combinations/onemodel/"+  m + "/" + str(i) + "/"	
#       os.makedirs(out_dir, exist_ok=True)
#       pr_auc = combine_results(
#       "/results",
#       [m],
#       prompts,
#       i,
#       "predictions.csv",
#       out_dir,
#       sample=1
#    )

# for p in prompts:
#    for i in range(9,10):
#       out_dir = "result_combinations/oneprompt/"+ p + "/" + str(i) + "/"	
#       os.makedirs(out_dir, exist_ok=True)
#       pr_auc = combine_results(
#       "/results",
#       models,
#       [p],
#       i,
#       "predictions.csv",
#       out_dir,
#       sample=1
#    )

# models = [
#           "llama31-8B", 
#           "llama-32-3B", 
#           "mistral-7B", 
#           "mistral-7B-0p3",
#           "openhermes-2p5-7B", 
#           "qwen2-7B", 
#           "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]
# for i in range(15,121, 15):
#     out_dir = "result_combinations/allm_allp/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=100
#     )

# models = [
#           "llama31-8B", 
#           "llama-32-3B", 
#           "mistral-7B", 
#           "mistral-7B-0p3",
#           "openhermes-2p5-7B", 
#         #   "qwen2-7B", 
#         #   "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]
# for i in range(45, 62, 15):
#     out_dir = "result_combinations/all_wo_qwen_teuken_NEW/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=100
#     )

# models = [
#         #   "llama31-8B", 
#         #   "llama-32-3B", 
#         #   "mistral-7B", 
#         #   "mistral-7B-0p3",
#         #   "openhermes-2p5-7B", 
#         #   "qwen2-7B", 
#         #   "teuken-7B-0p4", 
#           "llama31-70B", 
#           "mistral-8x7B-0p1"
#           ]
# prompts = [
#            "alltypes-de-abstitle-8-0", 
#            "highlemma-fewlabels",
#            "highlemma-manylabels",
#            "lowlemma-fewlabels",
#            "lowlemma-manylabels", 
#            "alltypes-de-abstitle-8-1", 
#            "alltypes-de-abstitle-8-2", 
#            "alltypes-de-abstitle-8-3",
#            "alltypes-de-abstitle-8-4",
#            "english-0-8",
#            "english-1-8",
#            "english-2-12",
#            "mixed-0-8",
#            "mixed-1-8",
#            "mixed-2-12"
#            ]
# for i in range(45, 62, 15):
#     out_dir = "result_combinations/all_wo_qwen_teuken_NEW/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=100
#     )

models = [
          "llama31-8B", 
          "llama-32-3B", 
          "mistral-7B", 
          "mistral-7B-0p3",
          "openhermes-2p5-7B", 
          "qwen2-7B", 
          "teuken-7B-0p4", 
          "llama31-70B", 
          "mistral-8x7B-0p1"
          ]
prompts = [
           "alltypes-de-abstitle-8-0", 
           "highlemma-fewlabels",
        #    "highlemma-manylabels",
        #    "lowlemma-fewlabels",
           "lowlemma-manylabels", 
        #    "alltypes-de-abstitle-8-1", 
        #    "alltypes-de-abstitle-8-2", 
        #    "alltypes-de-abstitle-8-3",
        #    "alltypes-de-abstitle-8-4",
        #    "english-0-8",
        #    "english-1-8",
           "english-2-12",
        #    "mixed-0-8",
        #    "mixed-1-8",
           "mixed-2-12"
           ]
for i in range(38,39):
    out_dir = "result_combinations/selectedprompts_3bigmodelexps/" + str(i) + "/"	
    os.makedirs(out_dir, exist_ok=True)
    pr_auc = combine_results(
    "/results",
    models,
    prompts,
    i,
    "predictions.csv",
    out_dir,
    sample=50,
    max_bigmodel=3
    )
# for i in range(5,6):
#     out_dir = "result_combinations/selectedprompts_restrictedbigmodels/" + str(i) + "/"	
#     os.makedirs(out_dir, exist_ok=True)
#     pr_auc = combine_results(
#     "/results",
#     models,
#     prompts,
#     i,
#     "predictions.csv",
#     out_dir,
#     sample=1,
#     max_bigmodel=3
#     )