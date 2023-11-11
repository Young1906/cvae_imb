import os
import yaml


if __name__ == "__main__":
    with open("config/_dev_mcmc.yml", "r") as f:
        tmp = yaml.safe_load(f)

    ls_dataset = ["ionosphere", "ecoli", "frogs", "breast-tissue", "heart_2cl", "connectionist", "parkinsons", "balance", "breast-cancer"]
    ls_clf = ["svm", "lr", "knn", "decision_tree", "mlp", "gbc", "catboost"]

    ls_mcmc_d = [ "lr" ]


    for ds in ls_dataset:
        for clf in ls_clf:
            tmp["mcmc"]["dataset"] = ds
            tmp["mcmc"]["classifier"] = clf
            tmp["mcmc"]["score_avg_method"] = "macro"

            fn = f"{ds}_{clf}"
            with open(f"config/mcmc/{fn}.yml", "w") as f:
                yaml.dump(tmp, f)



