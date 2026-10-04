import yaml
from data_preparation import DataPreparation
from model_training import ModelTraining

def load_config(path="./src/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config_path = "./src/config.yaml"
    config = load_config(config_path)
    dp = DataPreparation(config_path)
    df = dp.load_data()
    X, y = dp.prepare_features(df)
    trainer = ModelTraining(config, dp.create_preprocessor())

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    comparison = trainer.train_all_models(X_train, y_train, X_val, y_val)

    print("\nMODEL COMPARISON - VALIDATION SET")
    print(comparison[[
        "Model", "CV_R2", "Validation_MAE",
        "Validation_RMSE", "Validation_R2"
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name, best_model = trainer.select_best_model()
    print(f"\nBEST MODEL: {best_name}")

    metrics = trainer.evaluate_final_model(
        best_model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print("\nFINAL TEST PERFORMANCE")
    print(f"Test MAE : {metrics['MAE']:.4f}")
    print(f"Test RMSE: {metrics['RMSE']:.4f}")
    print(f"Test R2  : {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
