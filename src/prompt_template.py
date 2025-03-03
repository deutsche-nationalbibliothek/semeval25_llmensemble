"""
File name: prompt_template.py
Description: Module to build few-shot prompts for LLMs by reading the prompt files and applying a template to them.
"""

import json
import pandas as pd
from ast import literal_eval


class PromptBuilder:
    """Class to parse and build prompts.

    Methods:
        __init__: Initialization.
        parse_prompt_file: Returns text-label-tupels from a prompt_file
        build_prompt_by_template: Builds a prompt to be input to an LLM.
    """

    def __init__(self, prompt_file, custom_instruction: str):
        """Initialization.

        Args:
            prompt_file (str): Path to the prompt file containing the few-shot examples.
            model_name (str): Identifier of a HuggingFace model.
            custom_instruction (str): Instruction for the prompt."""

        self.prompt_file = prompt_file
        self.doc_labels = self.parse_prompt_file(prompt_file)
        self.custom_instruction = custom_instruction

    def parse_prompt_file(self, prompt_file: str):
        """Returns text-label-tupels from a prompt_file.

        Is called by __init__.
        """

        print(prompt_file)
        texts_labels = []
        with open(prompt_file, encoding="utf-8") as promptf:
            current_text = ""
            current_labels = ""
            for line in promptf:
                if line.startswith("Extrahiere Schlagwörter"):
                    pass
                elif line.startswith("Text: "):
                    text = line[len("Text: ") :]
                    current_text = text.strip()
                elif line.startswith("Schlagwörter: "):
                    labels = line[len("Schlagwörter: ") :]
                    current_labels = labels.strip()
                    texts_labels.append([current_text, current_labels])
                    current_text = ""
                    current_labels = ""
                elif line.startswith("###"):
                    pass
        return texts_labels

    def build_prompt_by_template(self, template_file, debug=False):
        """Builds a prompt to be input to an LLM.

        Args:
            template_file (Path): Path to a template file (.json) containing
                                  patterns for the different prompt parts.
            debug (bool): If True, prints intermediate steps of the prompt building.
        Returns:
            prompt (str): A prompt with a slot to fill in the test text.
        """

        with open(template_file) as tf:
            template = json.load(tf)
        instruction = template["instruction"]
        example_pattern = template["example"]
        keywords_pattern = template["keywords"]
        test_item_pattern = template["test_item"]
        if debug:
            print("Instruction:", instruction)
            print("Example_pattern:", example_pattern)
            print("Keywords pattern:", keywords_pattern)
            print("Test item pattern:", test_item_pattern)
        if self.custom_instruction != "":
            prompt = instruction.format(custom_instruction=self.custom_instruction)
            if debug:
                print("Prompt after instruction: ", prompt)
        else:
            prompt = ""
        for text, keywords in self.doc_labels:
            prompt += example_pattern.format(text=text)
            if debug:
                print("Prompt after example: ", prompt)
            prompt += keywords_pattern.format(keywords=keywords)
            if debug:
                print("Prompt after keywords: ", prompt)
        prompt += test_item_pattern
        if debug:
            print("Final version of the prompt", prompt)
        return prompt


def sample_prompt(dataset_file, which_text, n_examples, outfile):
    """Samples examples for prompt combinations.

    Args:
        dataset_file (str): Dataset-file (csv) to sample from.
        which_text (str): Must be in ["title", "abstract", "both"].
                          Determines which text to use for the prompt.
        n_examples (int): Number of examples to sample.
        outfile (str): Path to the output file (.txt)
    """

    data = pd.read_csv(dataset_file)
    # columns: 'language', 'split', 'text_type', 'subjects', 'doc_id', 'title', 'abstract', 'dcterms:subject', 'labels'
    prompt_subset = data.sample(n_examples)

    urls = prompt_subset["doc_id"].tolist()
    prompt_texts = []
    if which_text == "title":
        prompt_texts = prompt_subset["title"].tolist()
    if which_text == "abstract":
        prompt_texts = prompt_subset["abstract"].tolist()
    elif which_text == "both":
        prompt_texts = list(
            zip(prompt_subset["title"].tolist(), prompt_subset["abstract"].tolist())
        )

    flattened_prompt_texts = []
    for pt in prompt_texts:
        if type(pt) == tuple:
            # both scenario
            both_text = ""
            for text_part in pt:
                if type(text_part) == list:
                    text_part = " ".join(text_part)
                both_text += text_part + " "
            flattened_prompt_texts.append(both_text)
        elif type(pt) == list:
            pt = " ".join(pt)
            flattened_prompt_texts.append(pt)
        else:
            flattened_prompt_texts.append(pt)

    prompt_labels = prompt_subset["labels"].tolist()
    eval_labels = [literal_eval(x) for x in prompt_labels]
    str_labels = [", ".join(x) for x in eval_labels]
    with open(outfile, "w") as f:
        for url, text, labels in zip(urls, flattened_prompt_texts, str_labels):
            f.write(f"IDN: {url}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Schlagwörter: {labels}\n")
            f.write("###\n")
