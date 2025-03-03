"""
File name: rank.py
Description: Implementation of rank-stage
"""
import yaml
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import json
import os
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from dataclasses import asdict
from dvc.api import params_show


def safe_int_conversion(rel):
    try:
        return int(rel)
    except (TypeError, ValueError):
        return 0

class Ranker:
    """Class to perform the ranking stage using vLLM.
    
    Methods:
        __init__: Initialize.
        rank: Starts the ranking and writes results.
        rank_batches: Performs the ranking over batches.
    """

    def __init__(
            self, 
            dataset_file: str, 
            predictions_file: str, 
            output_file: str
            ):
        """Initialiazes the Ranker class.
        
        Args:
            dataset_file (str): Path to the dataset file (csv). 
                                Needed to obtain the title/abstract of the texts.
            predictions_file (str): Path to the mapped predictions file (.arrow) from the mapping output.
            output_file (str): Arrow output file (for evaluation).
        """
        
        self.dvc_params = params_show()
        self.p_general = self.dvc_params["general"]
        self.vllm_engineargs = self.dvc_params["vllm"]["engineargs"]
        self.p_completion = self.dvc_params["completion"]
        self.p_ranking = self.dvc_params["ranking"]
        self.debug = self.p_general.get("debug", False)

        self.output_file = output_file

        self.global_samplingparams = self.dvc_params["vllm"]["global_samplingparams"]
        self.score_param = self.p_ranking.get("score_param", "relevance")
        self.ranking_model = self.p_ranking.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")           
        self.temperature = self.p_ranking.get("temperature",0)
        self.max_suggestions = self.p_ranking.get("max_suggestions", 10)    
        self.min_confidence = self.p_ranking.get("min_confidence", 0)
        self.max_confidence = self.p_ranking.get("max_confidence", 10)
        self.threshold_confidence = self.p_ranking.get("threshold_confidence", 0.5)
        
        self.vllm_samplingparams = SamplingParams(
            max_tokens = 5, # small number of tokens to generate, so to fit only one number
            min_tokens = 1,
            temperature= self.temperature,
            presence_penalty= self.global_samplingparams.get("presence_penalty", 0),
            frequency_penalty=self.global_samplingparams.get("frequency_penalty", 0),
            repetition_penalty=self.global_samplingparams.get("repetition_penalty", 1),
            top_p=self.global_samplingparams.get("top_p", 1),
            best_of=self.global_samplingparams.get("best_of", 1),
        )
        
        self.vllm_engineargs = EngineArgs(
            model=self.ranking_model, 
            download_dir=self.vllm_engineargs.get("download_dir", ".cache/huggingface/hub"),
            task=self.vllm_engineargs.get("task", "generate"),
            gpu_memory_utilization=self.vllm_engineargs.get("gpu_memory_utilization", 0.8), 
            tensor_parallel_size=self.vllm_engineargs.get("tensor_parallel_size", 2),
            dtype=self.vllm_engineargs.get("dtype", "float16")
        )
        # if necessary: restrict GPU usage to one specific device
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.llm = LLM(**asdict(self.vllm_engineargs))

        # Prepare prompt
        with open(self.p_ranking.get("instruction"), encoding='utf-8') as f:
            self.ranking_instruction = json.load(f)["instruction"]
        
        self.ranking_instruction = self.ranking_instruction.format(min_confidence=self.min_confidence, max_confidence=self.max_confidence)
        self.prompt_template_info = self.p_completion.get("prompt_template_info")
        self.prompt_template_dir = self.p_completion.get("prompt_template_dir")
        with open(self.prompt_template_info) as f:
            content = json.load(f)
            self.prompt_template_file = content[self.ranking_model] 
        with open(os.path.join(self.prompt_template_dir,self.prompt_template_file)) as f:
            template = json.load(f)
        self.prompt_frame = template["instruction"] + template["example"]
        self.prompt_frame = self.prompt_frame.format(custom_instruction=self.ranking_instruction, text="{text}")
    

        # Read dataset
        self.data = pd.read_csv(dataset_file)
        
        self.which_text = self.p_completion.get("text", "both")
        if self.which_text == "both":
            self.data["content"] = self.data["title"] + " " + self.data["abstract"]
        elif self.which_text == "title":
            self.data["content"] = self.data["title"]
        elif self.which_text == "abstract":
            self.data["content"] = self.data["abstract"]
        self.data = self.data[["doc_id","content"]]
        self.data = self.data.set_index("doc_id")
        
        if self.debug:
            print("Prompt frame: ", self.prompt_frame)
            print("Data: ")
            print(self.data.head())
            
        # Read predictions
        self.predictions = pd.read_csv(predictions_file)
        self.preds_for_ranking = self.predictions[["doc_id","term"]]
        self.preds_for_ranking = self.preds_for_ranking.groupby("doc_id").agg(list)
        if self.debug:
            print("Preds for ranking head")
            print(self.preds_for_ranking.head(5))
        self.preds_for_ranking = self.preds_for_ranking.join(self.data, on = "doc_id")
        # Find rows where the 'content' column is empty
        empty_content_rows = self.preds_for_ranking[self.preds_for_ranking['content'].isna()]
        empty_content_count = len(empty_content_rows)
        if self.debug:
            print(f"Number of rows with empty content: {empty_content_count}")
        if empty_content_count > 0:
            raise ValueError(f"Found {empty_content_count} rows with empty content.")
        if self.debug:
            print("Preds for ranking after join")
            print(self.preds_for_ranking.head())
        self.preds_for_ranking.columns = ["prelim_labels", "text"]
        if self.debug:
            self.preds_for_ranking = self.preds_for_ranking.head(5)
        self.pred_dict = {}
        self.predictions = self.predictions.set_index("doc_id")
        for doc_id, row in self.predictions.iterrows():
            if doc_id not in self.pred_dict:
                self.pred_dict[doc_id] = {}
            self.pred_dict[doc_id][row["term"]] = {"term": row["term"],
                                                    "count": row["count"],
                                                    "cosine_similarity": row["cosine_similarity"],
                                                    "label_id": row["label_id"],
                                                    "score": row["score"]}
        if self.debug:
            print("Predictions has this shape:", self.predictions.shape)
            keys_in_pred_dict = 0
            for key in self.pred_dict.keys():
                keys_in_pred_dict += len(self.pred_dict[key])
            print("Pred_dict has this shape:", keys_in_pred_dict)
            print("Pred_dict keys:", list(self.pred_dict.keys())[0:5])
            print("Pred_dict items: ", list(self.pred_dict.items())[0:5])
        del self.predictions

    def rank_batches(self, preds_for_ranking):
        """Rankes the predictions in batches."""
        results = []
        for doc_id, row in tqdm(preds_for_ranking.iterrows(), total=len(preds_for_ranking), desc="Reranking suggestions"):
            text = row["text"]
            labels = sorted(list(set(row["prelim_labels"])))
            prompts = []
            for l in labels:
                prompt = self.prompt_frame.format(text="Text: {}\nSchlagwort:{}".format(text, l))
                prompts.append(prompt)

            response = self.llm.generate(prompts, self.vllm_samplingparams)
            answers = [o.outputs[0].text for o in response] 
            assert len(answers) == len(labels), "Number of answers does not match number of labels"
            answers_ints = []
            for a in answers:
                only_digits = ''.join([c for c in a if c.isdigit()])
                if len(only_digits) == 0:
                    # no digits in the answer, so assume 0
                    only_digits = 0
                answers_ints.append(only_digits)
            ranked_kw = list(zip(labels, answers_ints))
                
            if self.debug:
                print("This is the current item\nText:{}\nLabels:{}\nranked_kw:{}\n".format(
                    text, labels, ranked_kw
                ))
                correctly_returned = [x for x in labels if x in [kw for kw, r in ranked_kw]]
                print("Returned {}/{} keywords".format(len(correctly_returned), len(labels)))

           
            # More ranked keywords than possible
            if self.debug:
                print("Hallucinated keywords: ", [kw for kw, r in ranked_kw if kw not in labels])
                print("Original keywords: ", labels)
                print("Ranking output: ", ranked_kw)
            ranked_kw = [(kw,r) for kw,r in ranked_kw if kw in labels] 
            

            accepted_kw = 0
            for keyword, relevance in ranked_kw:
                if float(relevance) >= self.threshold_confidence:
                    # enough relevance
                    results.append({"doc_id": doc_id, "keyword": keyword, "relevance": relevance})
                    accepted_kw += 1
                    if self.debug==True:
                        print("Appending item: ", str({"doc_id": doc_id, "keyword": keyword, "relevance": relevance}))
                if accepted_kw == self.max_suggestions:
                    break

        return results  
    
    def rank(self):
        """Starts the ranking and writes results"""
       
        prelim_results = self.rank_batches(self.preds_for_ranking)

        ranked_results = []
        for item in prelim_results:
                doc_id, term, relevance = item["doc_id"], item["keyword"], item["relevance"]
                res_entry = {"doc_id": doc_id, "term": term, "relevance": relevance}
                res_entry.update(self.pred_dict[doc_id][term])
                ranked_results.append(res_entry)

        ranked_results = pd.DataFrame(ranked_results)

        if self.debug:
            print(ranked_results)
            return
        if self.score_param == "score":
            # Keep the score from previous stages = do nothing
            pass
        else:
            ranked_results[self.score_param] = ranked_results[self.score_param].astype(float)
            ranked_results["score"] = ranked_results[self.score_param]/(ranked_results[self.score_param].max()) 
        ranked_results = ranked_results[["doc_id", "label_id", "score"]]
        ranked_results.columns = ["doc_id", "label_id", "score"]
        # make sure each tuple doc_id label_id is unique
        ranked_results = ranked_results.groupby(["doc_id", "label_id"]).agg({"score":"max"}).reset_index()
        ranked_results.to_csv(self.output_file)


def execute():
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", help="Dataset Filename/Path", type=str, required=True)
    parser.add_argument("--predictions_file", help="Predictions Filename/Path", type=str, required=True)
    parser.add_argument("--output_file", help="Output Filename/Path", type=str, required=True)
    
    args = parser.parse_args()
    ranker = Ranker(args.dataset_file, args.predictions_file, args.output_file)
    ranker.rank()



if __name__ == "__main__":
    execute()