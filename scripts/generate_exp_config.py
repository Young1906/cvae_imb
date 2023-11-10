import os
import yaml


if __name__ == "__main__":
    with open("config/dev_exp.yml", "r") as f:
        tmp = yaml.safe_load(f)

    ls_clf = ["svm", "lr", "knn", "decision_tree", "mlp", "gbc", "catboost"]
    ratios = [1, 20, 30, 40, 50, 60, 70, 80, 90, 100]


    for clf in ls_clf:
        for ratio in ratios:
            tmp["mcmc"]["classifier"] = clf
            tmp["mcmc"]["ratio"]=ratio

            # name and result path
            tmp["mcmc"]["result_pth"] = "results/mcmc_exp1.csv"
            tmp["logger"]["logger_name"] = "mcm_exp1"

            fn = f"{clf}_{ratio}"
            with open(f"config/exp1/{fn}.yml", "w") as f:
                yaml.dump(tmp, f)




