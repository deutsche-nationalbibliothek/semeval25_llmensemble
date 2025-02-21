import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import requests
from time import strftime
import time

# parallelise
from multiprocessing import Pool
from tqdm import tqdm


def vectorize_opensource(batch, host="8090"):
    pos, batch = batch

    batch = tqdm(batch.iterrows(),
                 position= pos,
                 total=len(batch),
                 leave=True)

    logging.basicConfig(
        filename="vectorize.log", format="%(asctime)s %(message)s", filemode="a"
    )

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    def get_emb(row):
        for i in range(100):
            embedding = None
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                }

            payload = json.dumps(
                {
                    "inputs": str(row.Phrase),
                }
            )
            
            # response = requests.request(
            #     "POST",
            #     "http://127.0.0.1:{}/embed".format(host), #"http://127.0.0.1:12345",#
            #     headers=headers,
            #     data=payload,
            # )
            # embedding = response.json()[0]
            # # logger.info("vectorize successfully: " + str(row.phrase))
            # # print("Break")
            try:
                
                response = requests.request(
                "POST",
                "http://127.0.0.1:{}/embed".format(host), #"http://127.0.0.1:12345",#
                headers=headers,
                data=payload,
            )
                embedding = response.json()[0]
                # logger.info("vectorize successfully: " + str(row.phrase))
                # print("Break")
                break
            except Exception as e:
                print("emb at round i {}:".format(i), embedding)
                print(e)
                embedding = "Error"
                time.sleep(3)
                # logger.error(
                #     "Error\t" + 
                #     str(row.phrase)
                #     + "\t"
                #     + str(response.status_code)
                #     + "\t"
                #     + str(response.reason)
                #     + "\t"
                #     + str(response.text)
                # )
                    # raise e
        return embedding
    
    results = []
    for i, row in batch:
        emb = get_emb(row)
        if type(emb)==list:
            results.append({"gnd_idn": row.Code,
                            "term": row.Name,
                            "phrase": row.Phrase,
                            "vector":emb})
        else:
            print("Emb has type {}", type(emb))
            print("Emb: ", emb)
            raise ValueError
    return pd.DataFrame(results)
        


def process_opensource(input, output, host, phrase):
    print("Processing open source")
    with open(input) as f:
        df_gnd = pd.DataFrame(json.load(f))
    
    df_gnd = df_gnd[["Code","Name"]]#.head(20)
    # print(df_gnd, df_gnd.shape)
    df_gnd["Phrase"] = (
        phrase + df_gnd["Name"].astype(str)
    )
    # print("after filter", df_gnd.head())

    # print("after filter", df_gnd)
    # df_gnd = df_gnd.head(5000)

    # # parallelise
    # df_gnd = dd.from_pandas(df_gnd, npartitions=20)
    print("SHAPE", df_gnd.shape)
    if df_gnd.shape[0] < 205_000:
        print("Small dataset")
        batches = enumerate(np.array_split(df_gnd, 10))
        with Pool(processes = 10) as pool:
            result = pool.map(vectorize_opensource, batches)
        results_df = pd.concat(result)
        print(results_df.head())
        results_df.to_csv(output, index = False, mode="a", header=not os.path.exists(output))
    else:
        print("Dataframe too large, splitting into smaller chunks")
        df_gnd = df_gnd.reset_index(drop=True)
        for i in range(0, df_gnd.shape[0], 100_000):
            print("Processing next 100k at ", strftime("%H:%M:%S"))
            batch = df_gnd.iloc[i:i+100_000]
            batches = enumerate(np.array_split(batch, 20))
            with Pool(processes = 20) as pool:
                result = pool.map(vectorize_opensource, batches)
            results_df = pd.concat(result)
            print(results_df.head())
            results_df.to_csv(output, index = False, mode="a", header=not os.path.exists(output))


def run_opensource():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input Filename/Path", type=str, required=True)
    parser.add_argument(
        "--output", help="Output Filename/Path", type=str, required=True
    )
    parser.add_argument("--host", help="Host", type=str, default='8080')
    parser.add_argument("--phrase", help="Phrase", type=str, default="Ein gutes Schlagwort für dieses Dokument lautet: ")
    parser.add_argument("--labelkind", help="Labelkind", type=list, default=["prefLabel"])
    args = parser.parse_args()
    inp = args.input
    outp = args.output
    process_opensource(inp, outp, args.host, args.phrase)
    # print("Next 100k done at ", strftime("%H:%M:%S"))


run_opensource()


# Commands run: 
#python src/get_embeddings.py --input vocab/gnd.csv --output vocab/gnd_embeddings_0612.csv --host "8090" --phrase "" 
# python src/get_embeddings_parallel.py --input vocab/gnd.csv --output vocab/gnd_embeddings_test.csv --host "8090" --phrase "" 
# python src/get_embeddings_parallel.py --input vocab/gnd.csv --output vocab/gnd_embeddings_0617.csv --host "8090" --phrase "Ein gutes Schlagwort für dieses Dokument lautet: " 
# python src/get_embeddings_parallel.py --input vocab/gnd.csv --output vocab/gnd_embeddings_0617.csv --host "8090" --phrase "Ein gutes Schlagwort für dieses Dokument lautet: " 
# python src/get_embeddings_parallel.py --input vocab/gnd.csv --output ~/data/embeddings/gnd_embeddings_bgem3_full.csv --host "8090" --phrase "Ein gutes Schlagwort für dieses Dokument lautet: " --labelkind []
# python src/get_embeddings_parallel.py --input llms4subjects/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json --output assets/embeddings/embeddings-tib-core-phrase.csv --host "8090" --phrase "Ein gutes Schlagwort für dieses Dokument lautet: " 