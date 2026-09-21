# python eval/classify.py [--checkpoint PATH] [--per-class N] [--guidance W]

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

from eval.fad import (add_gen_args, class_names, gen_cache_path, get_device,
                      load_generated, load_meta, load_real, save_json)

FOLDS = 5


def coarse(names, idxs):
    return np.array([names[i].rsplit("_sub", 1)[0] for i in idxs])


# A linear probe on CLAP embeddings of real (vocoded) clips. Its cross-validated
# accuracy on real clips is the ceiling generated accuracy is read against
def fit_classifier(real):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    cv_pred = cross_val_predict(clf, real["emb"], real["cls"], cv=cv)
    clf.fit(real["emb"], real["cls"])
    return clf, cv_pred


def accuracy_report(clf, cv_pred, real, fake, names):
    pred = clf.predict(fake["emb"])
    rows = []
    for c in np.unique(fake["cls"]):
        r, f = real["cls"] == c, fake["cls"] == c
        wrong = Counter(pred[f][pred[f] != c]).most_common(1)
        rows.append({"key": names[c],
                     "real_acc": float((cv_pred[r] == c).mean()),
                     "fake_acc": float((pred[f] == c).mean()),
                     "fake_coarse_acc": float((coarse(names, pred[f]) ==
                                               coarse(names, [c])[0]).mean()),
                     "top_confusion": names[wrong[0][0]] if wrong else ""})
    return {"per_class": rows,
            "real_acc": float((cv_pred == real["cls"]).mean()),
            "real_balanced_acc": float(np.mean([r["real_acc"] for r in rows])),
            "fake_acc": float((pred == fake["cls"]).mean()),
            "fake_coarse_acc": float((coarse(names, pred) == coarse(names, fake["cls"])).mean())}


def print_accuracy(report, title):
    print("\n%s: subclass classifier accuracy" % title)
    print("%-16s %8s %8s %8s  %s" % ("subclass", "real(cv)", "fake", "coarse", "top confusion"))
    for row in report["per_class"]:
        print("%-16s %8.3f %8.3f %8.3f  %s" %
              (row["key"], row["real_acc"], row["fake_acc"], row["fake_coarse_acc"],
               row["top_confusion"]))
    print("real %.3f (balanced %.3f), fake %.3f, fake coarse %.3f, chance %.3f" %
          (report["real_acc"], report["real_balanced_acc"], report["fake_acc"],
           report["fake_coarse_acc"], 1.0 / len(report["per_class"])))


def main():
    parser = argparse.ArgumentParser(description="Class-conditional accuracy of "
                                                 "generated samples.")
    add_gen_args(parser)
    args = parser.parse_args()

    device = get_device()
    labels, stats = load_meta()
    names = class_names(labels)
    real = load_real(labels, device)
    gen = load_generated(args, labels, stats, device)

    clf, cv_pred = fit_classifier(real)
    report = accuracy_report(clf, cv_pred, real, gen, names)
    print_accuracy(report, "diffusion")
    save_json(report, "classify_%s" %
              os.path.basename(gen_cache_path(args)).replace(".npz", ".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
