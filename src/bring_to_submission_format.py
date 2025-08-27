"""
File name: bring_to_submission_format.py
Description: Script needed to convert output of the pipeline to the submission format for the LLMs4Subjects-shared task.
"""

import pandas as pd
import json
import os
from argparse import ArgumentParser


def create_submission_dir(predictions_file, dataset_file, general_out_dir, debug=False):
    preds = pd.read_csv(predictions_file)
    # preds = preds.rename({"doc_id": "idn"}, axis="columns")
    dataset = pd.read_csv(dataset_file)
    # assert dataset and preds have the same set of idn values, if not print this out
    print(preds.columns, dataset.columns)
    # assert (
    #     len(set(dataset.doc_id) - set(preds.doc_id)) == 0
    # ), "Some idn values are missing in the predictions or dataset file"
    # remove assertion, with only one model + one prompt some idns might not have predictions
    ext_preds = pd.merge(preds, dataset, on="doc_id")

    # Group by 'idn' and sort by 'score'
    grouped = ext_preds.sort_values(by="score", ascending=False).groupby("doc_id")

    # Aggregate 'label_id' values and keep 'language' and 'text_type'
    result = grouped.agg(
        {"label_id": lambda x: list(x), "language": "first", "text_type": "first"}
    ).reset_index()

    if debug:
        result = result[
            result.doc_id.isin(
                [
                    "3A1653943785",
                    "3A1769713336",
                    "3A60144180X",
                    "3A1657293173",
                    "3A169427103X",
                ]
            )
        ]

    # columns left: idn, label_id, language, text_type
    # label_id is a sorted list of label_ids

    # iterate over the rows, extract language and text type.
    for _, row in result.iterrows():
        language = row["language"]
        text_type = row["text_type"]
        label_ids = row["label_id"]
        doc_id = row["doc_id"]
        file_dir = os.path.join(general_out_dir, "subtask_2", text_type, language)
        os.makedirs(file_dir, exist_ok=True)
        with open(os.path.join(file_dir, f"{doc_id}.json"), "w") as f:
            json.dump({"dcterms:subject": label_ids}, f, indent=4)


def execute():
    parser = ArgumentParser()
    parser.add_argument(
        "--predictions_file", help="Predictions file (.csv)", type=str, required=True
    )
    parser.add_argument(
        "--dataset_file",
        help="Dataset file containing the info about text_type and language (.csv)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--general_out_dir",
        help="Output directory where the submission files will be stored",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    create_submission_dir(
        args.predictions_file, args.dataset_file, args.general_out_dir, debug=False
    )


if __name__ == "__main__":
    execute()
