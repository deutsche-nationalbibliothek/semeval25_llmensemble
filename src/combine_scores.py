from argparse import ArgumentParser
import os
import pandas as pd


def combine_scores(
    predictions_withscore_file,
    predictions_withrelevance_file,
    weight_score,
    output_file,
):
    withscore = pd.read_csv(
        predictions_withscore_file
    )  # cols: doc_id, label_id, score, term, cosine_similarity, count
    withrelevance = pd.read_csv(
        predictions_withrelevance_file
    )  # cols: doc_id, label_id, score

    merged_df = pd.merge(
        withscore,
        withrelevance,
        how="outer",
        on=["doc_id", "label_id"],
        suffixes=("_ensemble", "_relevance"),
    )

    merged_df.fillna({"score_ensemble": 0}, inplace=True)
    merged_df.fillna({"score_relevance": 0}, inplace=True)

    merged_df["score"] = merged_df.apply(
        lambda row: row["score_ensemble"] * weight_score
        + row["score_relevance"] * (1 - weight_score),
        axis=1,
    )

    wrong_scores = merged_df.query("score < 0 or score > 1")
    assert len(wrong_scores) == 0, "Some scores are not between 0 and 1"
    merged_df.to_csv(output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ranking_input",
        help="File that would be input for ranking stage",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ranking_output",
        help="File that is output of the ranking stage",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--weight_score",
        help="Weight for the score, weight for the relevance is 1-weigth_score; Must be between 0 and 1",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--output_file", help="Output Filename/Path .arrow", type=str, required=True
    )
    args = parser.parse_args()
    assert 0 <= args.weight_score <= 1, "Weight score must be between 0 and 1"
    assert os.path.exists(args.ranking_input), "File does not exist"
    assert os.path.exists(args.ranking_output), "File does not exist"
    combine_scores(
        args.ranking_input, args.ranking_output, args.weight_score, args.output_file
    )
