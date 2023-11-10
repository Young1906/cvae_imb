import os
import yaml


if __name__ == "__main__":
    with open("config/dev-baseline-exp.yml", "r") as f:
        tmp = yaml.safe_load(f)

    ls_sampler = ["adasyn",
                  "kmean-smote",
                  "smotenn",
                  "svm-smote",
                  "instance-hardness-threshold",
                  "baseline"]

    ls_clf = ["svm", "lr", "knn", "decision_tree", "mlp", "gbc", "catboost"]
    ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    for sampler in ls_sampler:
        for clf in ls_clf:
            for ratio in ratios:
                tmp["baseline"]["sampler_name"] = sampler 
                tmp["baseline"]["classifier"] = clf
                tmp["baseline"]["ratio"]=ratio

                fn = f"{sampler}_{clf}_{ratio}"
                with open(f"config/exp-baseline/{fn}.yml", "w") as f:
                    yaml.dump(tmp, f)




