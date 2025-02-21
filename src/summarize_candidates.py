from argparse import ArgumentParser
import logging
import pandas as pd
from dvc.api import params_show

def execute():
    parser = ArgumentParser()
    parser.add_argument("input_files", help="Input files", type=str, nargs='+')
    parser.add_argument("--output_file", help="Output Filename/Path", type=str, required=True)
    # parser.add_argument("--output_file_eval", help="Output Filename/Path .arrow", type=str, required=True)
    args = parser.parse_args()
    n_files = len(args.input_files)
    print("Input files:", args.input_files)

    logging.basicConfig(filename="logs/summarize_candidates.log", level=logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s')
    df = None
    for file in args.input_files:
        new_input = pd.read_csv(file)
        new_input = new_input.sort_values(by=['doc_id', 'label_id', 'cosine_similarity'], ascending=[True, True, True])
        new_input = new_input.groupby(['doc_id', 'label_id']).agg({
            'cosine_similarity': 'max',
            'candidate': 'last',
            'term': 'last'}).reset_index()

        logging.info(f"Processing file: {file}")
        if df is None:
          df = new_input
        else:
          df = pd.concat([df, new_input])

    df['score'] = df['cosine_similarity'] 

    df['score'] = df['score'].clip(lower=0, upper=1)

    df_grouped = df.groupby(['doc_id', 'label_id']).agg({
      'score': 'sum',
      'term': 'last',
      'cosine_similarity': 'max'
    }).reset_index()
    # Add a count column
    df_count = df.groupby(['doc_id', 'label_id']).size().reset_index(name='count')
    # Merge the count column with the grouped DataFrame
    df_grouped = df_grouped.merge(df_count, on=['doc_id', 'label_id'])

    df_grouped['score'] = df_grouped['score'] / n_files
    df_grouped.to_csv(args.output_file, index=False)
    
if __name__ == "__main__":
    execute()
