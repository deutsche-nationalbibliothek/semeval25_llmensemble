import pandas as pd 
from collections import defaultdict

def analyse_results(predictions_file, gold_standard_file, output_file):
    predictions = pd.read_csv(predictions_file) # columns idn,candidate,size,term,similarity,gnd_idn
    gold_standard = pd.read_csv(gold_standard_file) # columns language,split,text_type,subjects,url,title,abstract,dcterms:subject,labels
    predictions_readable_dict = defaultdict(list)
    predictions_idn_dict = defaultdict(list)
    for i, row in predictions.iterrows():
        predictions_readable_dict[row["idn"]].append(row["term"])
        predictions_idn_dict[row["idn"]].append(row["gnd_idn"])
    gold_standard["predicted_terms"] = gold_standard["url"].apply(lambda x: predictions_readable_dict[x])
    gold_standard["predicted_ids"] = gold_standard["url"].apply(lambda x: predictions_idn_dict[x])
    gold_standard = gold_standard[["url", "title", "dcterms:subject", "predicted_ids", "labels", "predicted_terms"]]
    gold_standard.to_csv(output_file, index=False)


analyse_results("/home/lisa/repos/semeval/results/predictions.csv", "/home/lisa/repos/semeval/datasets/tib-core-subjects-Article-Book-Conference-Report-Thesis-de-dev.csv", "/home/lisa/repos/semeval/results/analysis.csv")