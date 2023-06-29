import argparse
import pandas as pd
from operator import itemgetter
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import numpy as np
from utils.util import ptb_tokenize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_file", type=str)
    parser.add_argument("-r", "--reference_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()
    prediction_df = pd.read_json(args.prediction_file)
    # [n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    # [n, 5, var_len]
    captions = open(args.reference_file, "r").read().strip().split("\n")
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split("\t")
        row[0] = row[0][: len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider()]
    reference = key_to_refs


    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)

    output = {"SPIDEr": 0}
    outputs = {"SPIDEr": np.zeros(len(key_to_pred))}
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                    best_idx = max(enumerate(scores[n]), key=itemgetter(1))[0]
                    worst_idx = min(enumerate(scores[n]), key=itemgetter(1))[0]
                    print(f"Best {method}-{n+1} example: {prediction_df.iloc[best_idx]['prediction']}", file=writer)
                    print(f"Worst {method}-{n+1} example: {prediction_df.iloc[worst_idx]['prediction']}", file=writer)
                    print(f"Best {method}-{n+1} reference: {reference[prediction_df.iloc[best_idx]['img_id']]}", file=writer)
                    print(f"Worst {method}-{n+1} reference: {reference[prediction_df.iloc[worst_idx]['img_id']]}", file=writer)
                    
                    
            else:
                print(f"{method}: {score:.3f}", file=writer)
                best_idx = max(enumerate(scores), key=itemgetter(1))[0]
                worst_idx = min(enumerate(scores), key=itemgetter(1))[0]
                print(f"Best {method} example: {prediction_df.iloc[best_idx]['prediction']}", file=writer)
                print(f"Worst {method} example: {prediction_df.iloc[worst_idx]['prediction']}", file=writer)
                print(f"Best {method}-{n+1} reference: {reference[prediction_df.iloc[best_idx]['img_id']]}", file=writer)
                print(f"Worst {method}-{n+1} reference: {reference[prediction_df.iloc[worst_idx]['img_id']]}", file=writer)
            if method in ["CIDEr", "SPICE"]:
                output["SPIDEr"] += score
                outputs["SPIDEr"] += np.array(scores)
        output["SPIDEr"] /= 2
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)
        outputs["SPIDEr"] /= 2
        best_idx = max(enumerate(outputs["SPIDEr"]), key=itemgetter(1))[0]
        worst_idx = min(enumerate(outputs["SPIDEr"]), key=itemgetter(1))[0]
        print(f"Best SPIDEr example: {prediction_df.iloc[best_idx]['prediction']}", file=writer)
        print(f"Worst SPIDEr example: {prediction_df.iloc[worst_idx]['prediction']}", file=writer)
        print(f"Best {method}-{n+1} reference: {reference[prediction_df.iloc[best_idx]['img_id']]}", file=writer)
        print(f"Worst {method}-{n+1} reference: {reference[prediction_df.iloc[worst_idx]['img_id']]}", file=writer)
