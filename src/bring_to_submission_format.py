import pandas as pd
import json
import os
from argparse import ArgumentParser

def create_submission_dir(predictions_file, dataset_file, general_out_dir, debug = False):
    preds = pd.read_feather(predictions_file)
    preds = preds.rename({"doc_id":"idn"}, axis="columns")
    dataset = pd.read_csv(dataset_file)
    # assert dataset and preds have the same set of idn values, if not print this out
    print(preds.columns, dataset.columns)
    assert len(set(dataset.idn) - set(preds.idn)) == 0, "Some idn values are missing in the predictions or dataset file"
    ext_preds = pd.merge(preds, dataset, on='idn')

    # Group by 'idn' and sort by 'score'
    grouped = ext_preds.sort_values(by='score', ascending=False).groupby('idn')

    # Aggregate 'label_id' values and keep 'language' and 'text_type'
    result = grouped.agg({
        'label_id': lambda x: list(x),
        'language': 'first',
        'text_type': 'first'
    }).reset_index()

    if debug:
        result = result[result.idn.isin(['3A1653943785', '3A1769713336', '3A60144180X', '3A1657293173', '3A169427103X'])]


    # columns left: idn, label_id, language, text_type
    # label_id is a sorted list of label_ids
        
    # iterate over the rows, extract language and text type.
    for _ , row in result.iterrows(): 
        language = row['language']
        text_type = row['text_type']
        label_ids = row['label_id']
        idn = row['idn']
        file_dir = os.path.join(general_out_dir, text_type, language)
        os.makedirs(file_dir, exist_ok=True)
        with open(os.path.join(file_dir, f'{idn}.json'), 'w') as f:
            json.dump({'dcterms_subject': label_ids}, f, indent=4)

def execute():
    parser = ArgumentParser()
    parser.add_argument("--predictions_file", help="Predictions file (.arrow)", type=str, required=True)
    parser.add_argument("--dataset_file", help="Dataset file containing the info about text_type and language (.csv)", type=str, required=True)
    parser.add_argument("--general_out_dir", help="Output directory where the submission files will be stored", type=str, required=True)
    args = parser.parse_args()
    create_submission_dir(args.predictions_file, args.dataset_file, args.general_out_dir, debug = False)

if __name__ == "__main__":
    execute()

