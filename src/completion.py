"""
File name: completion.py
Description: Script to complete the complete-stage of our pipeline.
"""

import json
import datetime
import logging
import os
import pandas as pd
import logging
import datetime
from tqdm import tqdm

from pathlib import Path
import os
from transformers import AutoTokenizer
from argparse import ArgumentParser
from dataclasses import asdict

from prompt_template import PromptBuilder


from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.distributed.parallel_state import destroy_model_parallel

import torch
import gc

from dvc.api import params_show

# from huggingface_hub import login

# login(token=.os.environ["HF_TOKEN"])
# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


class LLMCompletion:
    """Class to perform the completion stage using vLLM.

    Methods:
        __init__: Initialize.
        predict: Starts the prediction and writes results.
        predict_batch: Predicts the completion for a batch of documents.
        post_process_results: Post-process the result returned by vLLM.
    """

    def __init__(
        self,
        hf_model_name: str,
        dataset_file: str,
        prompt_specification: str,
        completion_file: str = "completion.csv",
    ):
        """Initializes the LLMCompletion class.

        Args:
            hf_model_name (str): Identifier of a HuggingFace model.
            dataset_file (str): Path to the csv-dataset file, created with preprocess.py-script.
            prompt_specification (str): Path to the prompt file containing the few-shot examples.
            completion_file (str): Desired filename of the completion file.
                                   Defaults to completions.csv"
        """

        self.dvc_params = params_show()

        self.p_general = self.dvc_params["general"]
        self.vllm_engineargs = self.dvc_params["vllm"]["engineargs"]
        self.p_completion = self.dvc_params["completion"]
        self.completion_samplingparams = self.dvc_params["completion"]["vllm"][
            "samplingparams"
        ]
        self.global_samplingparams = self.dvc_params["vllm"]["global_samplingparams"]
        self.instruction_file = self.p_completion.get("custom_instruction")

        self.hf_model_name = hf_model_name
        self.vllm_engineargs = EngineArgs(
            model=self.hf_model_name,
            download_dir=self.vllm_engineargs.get(
                "download_dir", ".cache/huggingface/hub"
            ),
            task=self.vllm_engineargs.get("task", "generate"),
            gpu_memory_utilization=self.vllm_engineargs.get(
                "gpu_memory_utilization", 0.8
            ),
            tensor_parallel_size=self.vllm_engineargs.get("tensor_parallel_size", 2),
            dtype=self.vllm_engineargs.get("dtype", "float16"),
            trust_remote_code=True,
            enforce_eager=self.vllm_engineargs.get("enforce_eager"),
            max_model_len=(
                self.vllm_engineargs.get("max_model_len", 15000)
                if self.hf_model_name
                in [
                    "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                ]
                else None
            ),
        )

        self.vllm_samplingparams = SamplingParams(
            min_tokens=self.completion_samplingparams.get("min_new_tokens", 1),
            max_tokens=self.completion_samplingparams.get("max_new_tokens", 64),
            temperature=self.completion_samplingparams.get("temperature", 0),
            presence_penalty=self.global_samplingparams.get("presence_penalty", 0),
            frequency_penalty=self.global_samplingparams.get("frequency_penalty", 0),
            repetition_penalty=self.global_samplingparams.get("repetition_penalty", 1),
            top_p=self.global_samplingparams.get("top_p", 1),
        )

        self.llm = LLM(**asdict(self.vllm_engineargs))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        self.debug = self.p_general.get("debug", False)

        self.write_dir = Path(self.p_general.get("write_directory", None))
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

        self.splits = self.p_general.get("split", ["train", "dev"])
        self.text_types = self.p_general.get(
            "text_types", ["Article", "Book", "Conference", "Report", "Thesis"]
        )
        self.subjects = self.p_general.get(
            "subjects", ["all-subjects", "tib-core-subjects"]
        )
        self.languages = self.p_general.get("languages", ["en", "de"])
        self.which_text = self.p_completion.get("text", "both")
        self.n_processes = self.p_completion.get("n_processes", 1)
        self.batch_size = self.p_completion.get("batch_size", 100)

        try:
            self.data = pd.read_csv(dataset_file)
            print("Successfully loaded dataset with shape ", self.data.shape)
        except FileNotFoundError:
            print(
                "Dataset file not found... expected to read in data in this path: ",
                dataset_file,
            )
            raise FileNotFoundError

        self.completion_file = completion_file
        self.max_new_tokens = self.completion_samplingparams.get("max_new_tokens")
        self.max_total_tokens = self.completion_samplingparams.get("max_total_tokens")

        with open(self.instruction_file, encoding="utf-8") as f:
            self.custom_instruction = f.read()
        self.prompt_file = prompt_specification
        self.prompt_builder = PromptBuilder(self.prompt_file, self.custom_instruction)
        self.prompt_template_info = self.p_completion.get("prompt_template_info")
        self.prompt_template_dir = self.p_completion.get("prompt_template_dir")
        with open(self.prompt_template_info) as f:
            content = json.load(f)
            self.prompt_template_file = content[self.hf_model_name]
        self.prompt_frame = self.prompt_builder.build_prompt_by_template(
            os.path.join(self.prompt_template_dir, self.prompt_template_file)
        )
        self.prompt_length = len(self.tokenizer(self.prompt_frame)["input_ids"])
        self.free_tokens = (
            self.max_total_tokens - self.max_new_tokens - self.prompt_length
        )

        # get desired text: title/abstract/both
        if self.which_text == "both":
            self.data["content"] = self.data["title"] + " " + self.data["abstract"]
        elif self.which_text == "title":
            self.data["content"] = self.data["title"]
        elif self.which_text == "abstract":
            self.data["content"] = self.data["abstract"]
        else:
            raise ValueError(
                "Invalid text type. Choose from 'title', 'abstract', 'both'"
            )

        if self.debug:
            print("Completion file: ", self.completion_file)
            print("Splits: ", self.splits)
            print("Text types: ", self.text_types)
            print("Subjects: ", self.subjects)
            print("Languages: ", self.languages)
            print("Data: ", self.data.head())
            print("Prompt frame", self.prompt_frame)
            print("Free tokens: ", self.free_tokens)

    def post_process_results(self, response, prompt, doc_id):
        """Post-process the result returned by vLLM.

        This method is called by predict_batch.
        """

        answer_text = response.outputs[0].text
        answer = answer_text.split("\n")[0].split("<")[0]
        result = list(
            set([x.strip() for x in answer.split(", ") if len(x.strip()) > 1])
        )
        if self.debug:
            print("Result:", result)

        if len(result) == 0:
            logging.info("Results for document {} entirely empty".format(doc_id))
            result = []
        if self.debug:
            print("Prompt: ", prompt)
            print("Response: ", response.outputs[0].text)
            print("Result: ", result)
            print("-------------")
        return result

    def predict_batch(self, documents):
        """Predicts the completion for a batch of documents.

        This method is called by the predict-method.
        """

        results = []
        prompts = []
        doc_ids = []
        for _, row in documents.iterrows():
            row_content = row.content
            tokens = self.tokenizer(row_content)["input_ids"][: self.free_tokens]
            chunk = self.tokenizer.decode(tokens)
            prompt = self.prompt_frame.format(text=chunk)
            prompts.append(prompt)
            doc_ids.append(row.doc_id)

        responses = self.llm.generate(prompts, self.vllm_samplingparams)
        candidate_lists = []
        for i, response in enumerate(responses):
            result = self.post_process_results(response, prompts[i], doc_ids[i])
            candidate_lists.append(result)
        j = 0
        for candidate_list in candidate_lists:
            for candidate in candidate_list:
                results.append({"doc_id": doc_ids[j], "candidate": candidate})
            j += 1
        results_df = pd.DataFrame(results)
        return results_df

    def predict(self):
        """Starts the prediction and writes results."""

        results = []
        df_documents = self.data
        if self.debug:
            df_documents = df_documents.head(10)

        logging.info(
            "Start computation at time {}".format(str(datetime.datetime.now()))
        )

        batches = [
            df_documents[i : i + self.batch_size]
            for i in range(0, df_documents.shape[0], self.batch_size)
        ]
        results = []
        for batch in tqdm(batches, desc="Processing batches"):
            candidates = self.predict_batch(batch)
            results.append(candidates)

        candidates = pd.concat(results, ignore_index=True)

        if not self.debug:
            candidates.to_csv(
                self.completion_file,
                index=False,
                mode="a",
                header=not os.path.exists(self.completion_file),
            )
        logging.info(
            "End complete computation at time {}".format(str(datetime.datetime.now()))
        )
        destroy_model_parallel()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        logging.info(
            "Done with prediction! Find the results at {}".format(self.completion_file)
        )


def execute():
    """Builds an ArgumentParser and runs the completion stage."""
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_file", help="Dataset Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--hf_model_name",
        help="Identifier of hugging face model i.e. domain/model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--completion_file", help="Completion Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--prompt_specification",
        help="Path to prompt specification file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    completer = LLMCompletion(
        args.hf_model_name,
        args.dataset_file,
        args.prompt_specification,
        args.completion_file,
    )

    log_file_path = Path("logs/LLMCompletion.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, filemode="w", level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")

    completer.predict()


if __name__ == "__main__":
    execute()
