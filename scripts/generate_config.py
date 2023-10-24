import os
import yaml


if __name__ == "__main__":
    ls_config = os.listdir("config")
    ls_config = [x for x in ls_config if "yml" in x]

    for clf in ["svm", "lr", "knn", "decision_tree", "mlp", "gbc", "catboost"]:
        for _cfg in ls_config:
            with open(f"config/{_cfg}", "r") as f:
                cfg = yaml.safe_load(f)

            # assign classifier
            cfg['oversampling']['classifier'] = clf
                
            # checking
            if cfg['oversampling']['n_class'] > 2:
                cfg['oversampling']['score_avg_method'] = 'weighted'
            else:
                    cfg['oversampling']['score_avg_method'] = 'binary'

            cfg['oversampling']['result_pth'] = 'results/all.csv'

            # dump
            _cfg = _cfg.split(".")[0]
            _cfg = _cfg.replace("-", "_")
            fn = f"{_cfg}_{clf}.yml"

            with open(f"config/cvae/{fn}", "w") as f:
                yaml.dump(cfg, f)






