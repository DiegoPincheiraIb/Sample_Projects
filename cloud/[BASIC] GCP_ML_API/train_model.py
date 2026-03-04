import joblib
import xgboost as xgb
from src.dataset import IrisDatasetHandler
from sklearn.metrics import accuracy_score


def main():
    dataset_handler = IrisDatasetHandler()
    dataset_handler.split_dataset()
    X_train, y_train = dataset_handler.data['X_train'], dataset_handler.data['y_train']
    X_test,  y_test  = dataset_handler.data['X_test'],  dataset_handler.data['y_test']
    
    model = xgb.XGBClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.1,
        eval_metric   = "mlogloss",
        random_state  = 42,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(model, "iris_model.pkl")
    print("Model saved to iris_model.pkl")


if __name__ == "__main__":
    main()