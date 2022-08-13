import argparse
import json
from collections import Counter
import pickle
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression


def main(model_path: str, input_files: List[str]) -> None:
    data = []
    for filename in input_files:
        with open(filename) as f:
            data.extend(json.loads(f.read()).values())
    models = {}
    for train_label in ["bark", "whine"]:
        samples = []
        labels = []
        for frame in data:
            samples.append(frame["data"])
            labels.append(train_label in frame["labels"])
        if len(set(labels)) == 1:
            print("Skipping", train_label)
            continue
        X = np.array(samples, dtype=np.float64)
        y = np.array(labels, dtype=np.bool8)
        print(X.shape)
        print(y.shape)
        logreg = LogisticRegression(max_iter=2000)
        model = logreg.fit(X, y)
        print("Trained", train_label)
        print(model)
        #print(model.get_params())
        #print(model.coef_)
        #print(model.intercept_)
        print(Counter(zip(model.predict(X), y)))
        models[train_label] = model
    with open(model_path, "wb") as fb:
        pickle.dump(models, fb)


def pa() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=["data/2021-01-29.train.json"],
        required=True,
    )
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = pa()
    main(args.model_path, args.input_files)
