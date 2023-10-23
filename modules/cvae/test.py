import yaml
from modules.base import ModelConfig

if __name__ == "__main__":
    with open("config/dev.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config['model'])
    print(model_config)




