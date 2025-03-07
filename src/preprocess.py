"""
File name: preprocess.py
Description: This script prepares the dataset and is called part of the pipeline."""

import os
import json
import pandas as pd
import argparse
import re
import numpy as np


class DatasetCreator:
    """Class to create csv-datasets from the TIBKAT dataset.

    Methods:
        __init__: Initialization.
        get_data: Extracts the data from the TIBKAT dataset.
        write_dataset: Writes the dataset to a single file.
    """

    def __init__(
        self,
        data_path,
        data_splits,
        text_types,
        subjects,
        languages,
        sample_size,
        seed,
        forbidden_list=[],
        include_list=[],
    ):
        """Initialization.

        Args:
            data_path (str): Path to llms4subjects/shared-task-datasets/TIBKAT
            data_splits (list): List of data splits to consider (train, dev, test)
            text_types (list): List of text types to consider (Article, Book, Conference, Report, Thesis)
            subjects (str): Subjects to consider (all-subjects, tib-core-subjects)
            languages (list): Languages to consider (en, de)
            sample_size (int): Draw a subsample of size n from the dataset
            seed (int): Seed for the random sample
            forbidden_list (list) (Optional): List of forbidden doc_ids (that e.g. have been used in previous experiments)
            include_list (list) (Optional): List of doc_ids to include in the dataset (e.g. to re-create dev-test or dev-opt)
        """

        self.path = data_path  # point to llms4subjects/shared-task-datasets/TIBKAT
        self.sample_size = sample_size
        self.seed = seed
        for split in data_splits:
            if split not in ["train", "dev", "test"]:
                raise ValueError('data_split must be either "train" or "dev" or "test"')
        self.data_splits = sorted(data_splits)
        for tt in text_types:
            if tt not in ["Article", "Book", "Conference", "Report", "Thesis"]:
                raise ValueError(
                    'text_types must be a list containing only the following values: "Article", "Book", "Conference", "Report", "Thesis"'
                )
        self.text_types = sorted(text_types)
        for language in languages:
            if language not in ["en", "de"]:
                raise ValueError('language must be either "en" or "de"')
        self.languages = sorted(languages)
        for subject in subjects:
            print(subject)
            if subject not in ["all-subjects", "tib-core-subjects"]:
                raise ValueError(
                    'subjects must be in "all-subjects" or "tib-core-subjects"'
                )
        self.subjects = subjects
        self.forbidden_list = forbidden_list
        self.include_list = include_list
        self.data = self.get_data()

    def get_data(self):
        """Returns the data as a list.

        Returns:
            data (list): List of dictionaries, each with the following keys:
                - language: de|en
                - split: train|dev|test
                - text_type: Article|Book|Conference|Report|Thesis
                - subjects: all-subjects|tib-core-subjects
                - id: ID of the document
                - title: title of the document (potentially a list)
                - abstract: abstract of the document (potentially a list)
                - dcterms:subject: list of GND IDs
                - labels: list of GND terms
        """

        data = []
        url_string = "https://www.tib.eu/de/suchen/id/TIBKAT%"
        url_string_len = len(url_string)
        for subject in self.subjects:
            for split in self.data_splits:
                for text_type in self.text_types:
                    for language in self.languages:
                        current_wd = os.path.join(
                            self.path, subject, "data", split, text_type, language
                        )
                        try:
                            for p in os.listdir(current_wd):
                                pass
                        except FileNotFoundError:
                            print(f"Path {current_wd} does not exist. Skipping.")
                            continue
                        for p in os.listdir(current_wd):
                            with open(os.path.join(current_wd, p)) as f:
                                file_content = json.load(f)
                                if "@graph" in file_content:
                                    graph_nodes = file_content["@graph"]
                                    data_item = {}
                                    data_item["language"] = language
                                    data_item["split"] = split
                                    data_item["text_type"] = text_type
                                    data_item["subjects"] = subject

                                    label_to_term = {}
                                    label_texts = []
                                    for node in graph_nodes:
                                        try:
                                            node_id = node["@id"]
                                            if node_id.startswith("gnd:"):
                                                # Extract text from GND entity
                                                node_gndterm = node["sameAs"]
                                                label_to_term[node_id] = node_gndterm
                                            elif node_id.startswith("https:"):
                                                # Extract title, abstract, dcterms:subject, doc_id
                                                data_item["doc_id"] = node_id[
                                                    url_string_len:
                                                ]
                                                node_title = node["title"]
                                                if type(node_title) == list:
                                                    node_title = " ".join(node_title)
                                                data_item["title"] = node_title
                                                node_abstract = node["abstract"]
                                                if type(node_abstract) == list:
                                                    node_abstract = " ".join(
                                                        node_abstract
                                                    )
                                                data_item["abstract"] = node_abstract
                                                node_dcterms_subject = node[
                                                    "dcterms:subject"
                                                ]
                                                if type(node_dcterms_subject) == dict:
                                                    node_dcterms_subject = [
                                                        node_dcterms_subject
                                                    ]

                                                dcterms_subject = [
                                                    x["@id"]
                                                    for x in node_dcterms_subject
                                                ]
                                                for gnd_id in dcterms_subject:
                                                    if gnd_id in label_to_term:
                                                        label_texts.append(
                                                            label_to_term[gnd_id]
                                                        )
                                                data_item["dcterms:subject"] = (
                                                    dcterms_subject
                                                )

                                        except KeyError:
                                            pass
                                    data_item["labels"] = label_texts
                                    data.append(data_item)
        print("Data extraction of {} items completed.".format(len(data)))
        return data

    def write_dataset(self, output_file):
        """Writes the dataset obtained in get_data to a csv-file."""

        df = pd.DataFrame(self.data)
        df = df.drop_duplicates(subset=["doc_id"])

        old_size = df.shape[0]

        if len(self.forbidden_list) > 0:
            df = df[~df["doc_id"].isin(self.forbidden_list)]
            new_size = df.shape[0]
            print("Removed {} documents from the dataset.".format(old_size - new_size))

        if len(self.include_list) > 0:
            if self.sample_size is not None:
                print(
                    "Cannot use an include_list and sample at the same time. Exiting."
                )
                return
            else:
                df = df[df["doc_id"].isin(self.include_list)]
                new_size = df.shape[0]
                print("Included {} documents in the dataset.".format(new_size))

        if self.sample_size is not None:
            if self.sample_size > df.shape[0]:
                print(
                    "Sample size exceeds dataset size. Writing entire dataset to file instead"
                )
                df_final = df
            else:
                percentage = self.sample_size / df.shape[0]
                np.random.seed(self.seed)
                df_final = (
                    df.groupby(["subjects", "language", "text_type"])
                    .apply(lambda x: x.sample(frac=percentage, replace=False))
                    .reset_index(drop=True)
                )
        else:
            df_final = df
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_final.to_csv(output_file, index=False)
        print("Dataset of shape {} written to {}.".format(df_final.shape, output_file))


def run():
    """Runs the DatasetCreator get_data and write_dataset from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", help="Path to the data directory", type=str, required=True
    )
    parser.add_argument(
        "-s",
        "--data_splits",
        help="Data splits to consider. Can be passed as several strings or one string with - separators (e.g. train dev | train-dev)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--text_types",
        help="Text types to consider. Can be passed as several strings or one string with - separators (e.g. Article Book | Article-Book-Conference-Report-Thesis)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--languages",
        help="Languages to consider. Can be passed as several strings or one string with - separators (e.g. de en | de-en)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-u",
        "--subjects",
        help="Consider all-subjects or just tib-core-subjects (or both)? Should be passed as one string (e.g. all-subjects-tib-core-subjects)",
        default="all-subjects",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output file (csv, with relative path)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample_size",
        help="Draw a subsample of size n from the dataset",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--seed",
        help="Seed for the random sample",
        type=int,
        default=42,
        required=False,
    )
    parser.add_argument(
        "--forbidden_docs",
        help="Dataset with doc_ids that should not be included in this dataset",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--include_docs",
        help="Build a dataset with the documents in the include_docs dataset",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    subjects = []
    for subj in ["all-subjects", "tib-core-subjects"]:
        try:
            subjects.append(re.search(subj, args.subjects).group())
        except AttributeError:
            pass
    if subjects == []:
        raise ValueError(
            "subjects must include at least 1 out of (all-subjects, tib-core-subjects)"
        )
    splits = [x.split("-") for x in args.data_splits]
    splits = [item for sublist in splits for item in sublist]

    text_types = [x.split("-") for x in args.text_types]
    text_types = [item for sublist in text_types for item in sublist]

    languages = [x.split("-") for x in args.languages]
    languages = [item for sublist in languages for item in sublist]

    try:
        forbidden_docs_ds = pd.read_csv(args.forbidden_docs)
        # forbidden_list = forbidden_docs_ds.doc_id.tolist()
        forbidden_list = forbidden_docs_ds.idn.tolist()
        print("Using a forbidden_list of length {}.".format(len(forbidden_list)))
    except (FileNotFoundError, ValueError):
        print(
            "Not using a forbidden list (because the file was not found or the use was not requested)."
        )
        forbidden_list = []

    try:
        include_docs_ds = pd.read_csv(args.include_docs)
        include_list = include_docs_ds.idn.tolist()
        print("Using an include_list of length {}.".format(len(include_list)))
    except (FileNotFoundError, ValueError):
        include_list = []

    ds = DatasetCreator(
        data_path=args.data_path,
        data_splits=splits,
        text_types=text_types,
        subjects=subjects,
        languages=languages,
        sample_size=args.sample_size,
        seed=args.seed,
        forbidden_list=forbidden_list,
        include_list=include_list,
    )

    if args.output_file is not None:
        ds.write_dataset(output_file=args.output_file)


if __name__ == "__main__":
    run()
