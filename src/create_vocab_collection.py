"""
File name: create_vocab_collection.py
Description: Script for stage create_vocab_collection (prepares the embeddings for map-stage).
"""

import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import pyoxigraph
from pyoxigraph import RdfFormat
import pandas as pd
import torch
import weaviate
import weaviate.classes as wvc
from generate_embeddings import generate_embeddings
import unicodedata
import logging

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_vocab(
    ttl_path: Path, use_altLabels: bool = True, phrase: str = None
) -> pd.DataFrame:
    logging.info(f"Parsing vocabulary from {ttl_path}")
    with ttl_path.open("rb") as f:
        graph = pyoxigraph.parse(f, RdfFormat.TURTLE)
        labels: list[(str, str, bool)] = []

        for s, p, o, _ in tqdm(graph, desc="Processing triples"):
            uri = s.value
            label_id = uri.split("/")[-1]  # extract label_id from uri
            is_prefLabel = p.value == PREF_LABEL_IRI
            is_altLabel = p.value == ALT_LABEL_IRI
            label_text = o.value if phrase is None else f"{phrase}{o.value}"
            label_text = unicodedata.normalize("NFC", label_text)  # normalize unicode
            if is_prefLabel:
                labels.append((label_id, label_text, True))
            elif is_altLabel and use_altLabels:
                labels.append((label_id, label_text, False))

        labels = pd.DataFrame(
            labels, columns=["label_id", "label_text", "is_prefLabel"]
        )

        return labels


def create_collection(
    client: weaviate.Client,
    collection_name: str,
    overwrite: bool = False,
    TEI_port: str = "8090",
):
    logging.info(f"Attempting to create collection {collection_name} in Weaviate")
    if client.collections.exists(collection_name):
        logging.info(f"Collection {collection_name} already exists")
        if overwrite:
            client.collections.delete(collection_name)
            logging.info(f"Old Collection {collection_name} deleted")
        else:
            return collection_name

    client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="label_id",
                description="GND identifier for each concept",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.WORD,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="label_text",
                description="Label description (pref label or alt label)",
                data_type=wvc.config.DataType.TEXT,
                vectorize_property_name=False,
                tokenization=wvc.config.Tokenization.WORD,
                index_searchable=True,
                index_filterable=False,
            ),
            wvc.config.Property(
                name="is_prefLabel",
                description="Boolean: Label description is a SKOS prefLabel T/F",
                data_type=wvc.config.DataType.BOOL,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            ),
        ],
        # weaviate can connect to a huggingface text-embeddings service
        # so that queries can be vectorized on the fly
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_huggingface(
            # only works if huggingface TEI endpoint is running
            endpoint_url=f"http://host.docker.internal:{TEI_port}",
            vectorize_collection_name=False,
        ),
    )
    return collection_name


def insert_vocab(
    client: weaviate.Client,
    collection_name: str,
    vocab: pd.DataFrame,
    embeddings: torch.Tensor,
    phrase: str = None,
):
    logging.info(f"Inserting vocabulary into {collection_name}")
    this_collection = client.collections.get(collection_name)
    with this_collection.batch.dynamic() as batch:
        # Loop through the data
        for i, row in tqdm(vocab.iterrows()):

            # Build the object payload
            gnd_entity_obj = {
                "label_id": row["label_id"],
                "label_text": (
                    row["label_text"]
                    if phrase is None
                    else f"{phrase}{row['label_text']}"
                ),
                "is_prefLabel": row["is_prefLabel"],
            }

            # Add object to batch queue
            batch.add_object(properties=gnd_entity_obj, vector=embeddings[i].tolist())

    # Check for failed objects
    if len(this_collection.batch.failed_objects) > 0:
        logging.info(
            f"Failed to import {len(this_collection.batch.failed_objects)} objects"
        )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ttl_file", help="Vocab as SKOS ttl file", type=str, required=True
    )
    parser.add_argument(
        "--collection_name", help="Collection Name in Weaviate", type=str, required=True
    )
    parser.add_argument(
        "--TEI_port",
        help="Port on which Hugging Face text embedding service is running",
        type=str,
        default="8090",
    )
    parser.add_argument(
        "--phrase",
        help="Pass a context phrase to globally bias label embeddings",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing weaviate collection?",
        type=str,
        default=True,
    )
    parser.add_argument(
        "--arrow_out",
        help="Create a copy of the vocabulary in arrow format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_altLabels",
        help="Create embeddings for SKOS altLabels",
        type=str,
        default=True,
    )
    args = parser.parse_args()

    log_file_path = Path("logs/create_vocab_collection.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(message)s")

    vocab = parse_vocab(
        Path(args.ttl_file),
        use_altLabels=str2bool(args.use_altLabels),
        phrase=args.phrase,
    )

    client = weaviate.connect_to_local()
    if str2bool(args.overwrite):
        embeddings = generate_embeddings(vocab["label_text"].tolist())
        create_collection(
            client,
            args.collection_name,
            overwrite=str2bool(args.overwrite),
            TEI_port=args.TEI_port,
        )
        insert_vocab(client, args.collection_name, vocab, embeddings)
        # Note: phrase is already passed to parse_vocab

    if args.arrow_out is not None:
        vocab.to_feather(args.arrow_out)

    if not client.collections.exists(args.collection_name):
        raise ValueError(
            f"Collection {args.collection_name} does not exist. Try running with --overwrite=True"
        )
    client.close()


run()
