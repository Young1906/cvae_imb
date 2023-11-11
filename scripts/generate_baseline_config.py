import os
import yaml


if __name__ == "__main__":
    with open("config/dev-baseline.yml", "r") as f:
        tmp = yaml.safe_load(f)

    ls_dataset = ["ionosphere", "ecoli", "frogs", "breast-tissue", "heart_2cl", "connectionist", "parkinsons", "balance", "breast-cancer"]

    ls_clf = ["svm", "lr", "knn", "decision_tree", "mlp", "gbc", "catboost"]

    ls_sampler = ["adasyn",
                  "kmean-smote",
                  "smotenn",
                  "svm-smote",
                  "instance-hardness-threshold",
                  "baseline"]

    for ds in ls_dataset:
        for clf in ls_clf:
            for sampler in ls_sampler:
                tmp["baseline"]["dataset"] = ds
                tmp["baseline"]["classifier"] = clf
                tmp["baseline"]["sampler_name"] = sampler 
                tmp["baseline"]["score_avg_method"] = "macro"

                tmp["logger"]["name"] = "mcmc"
                tmp["logger"]["log_path"] = ".log"
                tmp["logger"]["telegram_handler"] = True

                fn = f"{ds}_{clf}_{sampler}"

                with open(f"config/baseline/{fn}.yml", "w") as f:
                    yaml.dump(tmp, f)
