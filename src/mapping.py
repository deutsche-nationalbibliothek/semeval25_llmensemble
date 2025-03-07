"""
File name: mapping.py
Description: Responsible for the map-stage of our pipeline.
"""

import json
import numpy as np
import pandas as pd
import requests
import datetime
import os
import yaml
import logging
from pathlib import Path
from typing import Any
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser
import time

from sklearn.metrics.pairwise import cosine_similarity
from weaviate.classes.query import MetadataQuery
import weaviate

from dvc.api import params_show


class Mapping:
    def __init__(
        self,
        hyperparameters: dict,
        collection_name: str,
        phrase: str,
        debug: bool,
        db_connection: weaviate.client.Client,
    ):
        self.hyperparameters = hyperparameters
        self.phrase = phrase
        self.use_phrase = self.hyperparameters.get("use_phrase", False)
        self.weaviate_client = db_connection
        assert self.weaviate_client.is_ready() == True, "Weaviate is not ready"
        self.collection_name = collection_name
        self.host = self.hyperparameters.get("host", "8090")
        self.debug = debug
        self.search = self.hyperparameters.get("search", "vector")
        self.alpha = self.hyperparameters.get("alpha", 0.5)

    def query_vector_database(self, candidate: str):
        chunks = self.weaviate_client.collections.get(self.collection_name)
        if self.use_phrase:
            query = self.phrase + candidate
        else:
            query = candidate

        max_retries = 10
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                embedding = list(
                    np.array(
                        requests.post(
                            "http://127.0.0.1:{}/embed".format(self.host),
                            headers={"Content-Type": "application/json"},
                            json={"inputs": query},
                        ).json()
                    ).reshape(-1)
                )
                if self.search == "vector":
                    response = chunks.query.near_vector(
                        near_vector=embedding,
                        limit=1,
                        return_metadata=MetadataQuery(distance=True),
                    ).objects[0]
                    result = {
                        candidate: {
                            "label_id": response.properties["label_id"],
                            "term": response.properties["label_text"],
                            "cosine_similarity": 1 - response.metadata.distance,
                            "hybrid_score": None,
                        }
                    }
                elif self.search == "hybrid":
                    # alternative: hybrid search
                    # alpha 0 = pure keyword search; alpha 1 = pure vector search
                    response = chunks.query.hybrid(
                        query=query,
                        vector=embedding,
                        limit=1,
                        return_metadata=MetadataQuery(score=True),
                        include_vector=True,
                        alpha=self.alpha,
                    ).objects[0]
                    result = {
                        candidate: {
                            "label_id": response.properties["label_id"],
                            "term": response.properties["label_text"],
                            "hybrid_score": response.metadata.score,
                            "cosine_similarity": np.dot(
                                embedding, response.vector["default"]
                            ),
                        }
                    }
                else:
                    raise ValueError(
                        "Invalid search type. Current options: vector, hybrid"
                    )
                success = True
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print("All attempts failed.")
                    success = False
                    result = {
                        candidate: {"label_id": [], "term": [], "cosine_similarity": []}
                    }
                    raise  # Re-raise the exception if all retries fail

        if success == False:
            print("Failed mapping with candidate: ", candidate)
        if self.debug:
            print("Query: ", query)
            print("Success: ", success)
            print("Result: ", result)
        return result


class LLMMapping:
    """Class to perform the mapping stage using Weaviate."""

    def __init__(
        self,
        completion_file: str,
        output_file: str,
        id_mapping_table: str,
        mapping_stats: str,
        allowed_subjects: str,
    ):
        """Initialization.

        Args:
            completion_file (str): output file of previous completion stage
            output_file (str): output file (csv) of the mapping stage
            id_mapping_table (str): optional mapping to a different vocabulary TODO: remove this for publication?
            mapping_stats (str): output file (json) for mapping statistics
            allowed_subjects (str): csv list of allowed subjects TODO: remove this for publication?
        """

        self.dvc_params = params_show("params.yaml")
        self.p_general = self.dvc_params["general"]
        self.p_mapping = self.dvc_params["mapping"]

        self.debug = self.p_general.get("debug", False)
        self.collection_name = self.p_general.get("collection_name", "GND")
        self.completion_file = completion_file
        self.predictions = pd.read_csv(self.completion_file)
        self.output_file = output_file
        self.id_mapping_table = id_mapping_table
        self.mapping_stats = mapping_stats
        self.allowed_subjects = allowed_subjects

        self.n_processes = self.p_mapping.get("n_processes", 20)
        self.phrase = self.p_mapping.get("phrase", None)
        self.score_param = self.p_mapping.get("score_param", "hybrid_score")
        assert self.score_param in [
            "cosine_similarity",
            "hybrid_score",
        ], "Invalid score parameter"
        self.hyperparameters = self.p_mapping.get("hyperparameters", {})
        self.search = self.hyperparameters.get("search", "vector")
        self.min_cosine_similarity = self.hyperparameters.get(
            "min_cosine_similarity", 0
        )

    def map_candidate_batches(self, batch):
        pos, batch = batch
        weaviate_client = weaviate.connect_to_local()
        assert weaviate_client.is_ready() == True, "Weaviate is not ready"

        results = []
        batch = tqdm(batch, position=pos, total=len(batch), leave=True)
        for b_item in batch:
            results.append(self.map_candidate(b_item, weaviate_client))

        weaviate_client.close()
        return results

    def map_candidate(self, candidate, db_connection):
        mapping = Mapping(
            hyperparameters=self.hyperparameters,
            collection_name=self.collection_name,
            phrase=self.phrase,
            debug=self.debug,
            db_connection=db_connection,
        )

        result = mapping.query_vector_database(candidate=candidate)
        return result

    def map(self):
        results = []
        df_predictions = self.predictions
        if self.debug:
            df_predictions = df_predictions.head(16)

        df_predictions = df_predictions.groupby(
            df_predictions.columns.tolist(), as_index=False
        ).size()
        # extract all different free candidates
        candidates = list(set(df_predictions.candidate.tolist()))
        logging.info(
            "Start computation at time {}".format(str(datetime.datetime.now()))
        )
        batches = enumerate(np.array_split(candidates, self.n_processes))
        with Pool(processes=self.n_processes) as pool:
            nested_result = pool.map(self.map_candidate_batches, batches)

        results = dict()
        for batch_result in nested_result:
            for indiv_result in batch_result:
                results.update(indiv_result)

        terms, cosine_similarities, hybrid_scores, label_ids = [], [], [], []

        for i, row in df_predictions.iterrows():
            candidate = row.candidate
            mapping = results[candidate]
            terms.append(mapping["term"])
            cosine_similarities.append(mapping["cosine_similarity"])
            hybrid_scores.append(mapping["hybrid_score"])
            label_ids.append(mapping["label_id"])

        df_predictions["term"] = terms
        df_predictions["cosine_similarity"] = cosine_similarities
        df_predictions["hybrid_score"] = hybrid_scores
        df_predictions["label_id"] = label_ids

        shape_before_filter = df_predictions.shape[0]
        removed_predictions = df_predictions[
            df_predictions.cosine_similarity < self.min_cosine_similarity
        ]
        if self.debug:
            print("Removed predictions:")
            print(removed_predictions.head(100))
        df_predictions = df_predictions[
            df_predictions.cosine_similarity >= self.min_cosine_similarity
        ]
        logging.info(
            "Number of predictions before and after filtering with threshold {}: {} -> {}".format(
                self.min_cosine_similarity, shape_before_filter, df_predictions.shape[0]
            )
        )

        if self.id_mapping_table:
            mapping_table = pd.read_csv(self.id_mapping_table)
            if not all(col in mapping_table.columns for col in ["idn", "gnd_id"]):
                raise ValueError(
                    "The mapping table must contain 'idn' and 'gnd_id' columns"
                )
            df_predictions = df_predictions.merge(
                mapping_table, left_on="label_id", right_on="idn", how="left"
            )

            # Rename columns to avoid conflicts
            df_predictions = df_predictions.rename(
                columns={"label_id": "dnbinternal_gnd_idn", "gnd_id": "label_id"}
            )
            df_predictions["label_id"] = df_predictions["label_id"].apply(
                lambda x: "gnd:" + str(x)
            )

            missing_terms = sum(df_predictions["label_id"].isna())
            if missing_terms > 0:
                logging.warning(
                    f"{missing_terms} internal gnd_idn values could not be mapped to gnd_id"
                )
                df_predictions = df_predictions.dropna(subset=["label_id"])
        else:  # when weaviate already return external gnd_id
            # add "gnd:" prefix to label_id to ensure consistency with TIBKAT
            df_predictions["label_id"] = df_predictions["label_id"].apply(
                lambda x: "gnd:" + str(x)
            )

        if self.allowed_subjects:
            allowed_subjects = pd.read_csv(self.allowed_subjects)
            n_before = df_predictions.shape[0]
            df_predictions = df_predictions.merge(
                allowed_subjects, left_on="label_id", right_on="nid", how="inner"
            )
            n_after = df_predictions.shape[0]

        df_predictions = df_predictions.sort_values(
            by=["doc_id", "label_id", "cosine_similarity"], ascending=[True, True, True]
        )
        df_predictions = (
            df_predictions.groupby(["doc_id", "label_id"])
            .agg(
                {
                    "candidate": "last",
                    "term": "last",
                    "cosine_similarity": "max",
                    "hybrid_score": "max",
                }
            )
            .reset_index()
        )

        df_predictions["score"] = df_predictions[self.score_param]

        # write mapping statistics to json file
        n_keywords = df_predictions.shape[0]
        n_matches_below_threshold = removed_predictions.shape[0]
        mapping_stats = {
            "n_keywords": n_keywords,
            "n_matches_below_threshold": n_matches_below_threshold,
            "n_not_in_allowed_subjects": (
                n_before - n_after if self.allowed_subjects else None
            ),
        }

        with open(self.mapping_stats, "w") as f:
            json.dump(mapping_stats, f)

        if self.debug:
            print(df_predictions.head())
        df_predictions.to_csv(
            self.output_file,
            index=False,
            mode="a",
            header=not os.path.exists(self.output_file),
        )
        path = os.path.dirname(self.output_file)
        removed_predictions.to_csv(
            os.path.join(path, "removed_" + os.path.basename(self.output_file)),
            index=False,
            mode="a",
            header=not os.path.exists(self.output_file.replace(".csv", "_removed.csv")),
        )

        logging.info("End computation at time {}".format(str(datetime.datetime.now())))


def execute():
    parser = ArgumentParser()
    parser.add_argument(
        "--completion_file", help="Completion Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--id_mapping_table",
        help="Optional mapping table which maps DNB internal idn to gnd-id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--mapping_stats", help="Statistics on Mapping", type=str, required=False
    )
    parser.add_argument(
        "--output_file", help="Output Filename/Path", type=str, required=True
    )
    parser.add_argument(
        "--allowed_subjects",
        help="Optional list of allowed identifiers to limit vocab",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    mapper = LLMMapping(
        args.completion_file,
        args.output_file,
        args.id_mapping_table,
        args.mapping_stats,
        args.allowed_subjects,
    )
    logging.basicConfig(filename="assets/logging/LLMMapping.log", level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")

    mapper.map()


if __name__ == "__main__":
    execute()
