import yaml
from src.data.load_data import load_dataset
from src.features.build_features import create_features
from src.models.train import train_model

def main(config_path='config/base_config.yaml'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    df = load_dataset(cfg['data_path'])
    X, y = create_features(df)
    model = train_model(X, y, cfg['model'])
    # Save model logic here

if __name__ == '__main__':
    main()
