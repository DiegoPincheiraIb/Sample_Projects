from src.dataset import WineDatasetHandler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

#! =============================================================================
N_COMPONENTS      = 2   # Number of principal components for PCA
N_ESTIMATORS      = 100 # Number of trees in the random forest
RANDOM_STATE      = 42  # Seed for reproducibility
TEST_SIZE         = 0.2 # Proportion of the dataset to include in the test split
#! =============================================================================

def main():
    # Load and prepare the dataset
    dataset_handler = WineDatasetHandler()
    num_samples, num_features, classes = dataset_handler.get_dataset_info()
    print(f"Dataset: {num_samples} samples, {num_features} features")
    print(f"Classes: {classes}")

    # Split the dataset into training and testing sets
    dataset_handler.split_dataset(test_size    = TEST_SIZE,
                                  random_state = RANDOM_STATE)

    # Create a pipeline with StandardScaler, PCA, and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components = N_COMPONENTS,
                       random_state = RANDOM_STATE)),
        ('rf',     RandomForestClassifier(n_estimators = N_ESTIMATORS,
                                          random_state = RANDOM_STATE))
    ])

    # Train the model
    pipeline.fit(dataset_handler.data['X_train'],
                 dataset_handler.data['y_train'])
    
    # Evaluate the model
    cv_score = pipeline.score(dataset_handler.data['X_test'],
                              dataset_handler.data['y_test'])
    y_pred   = pipeline.predict(dataset_handler.data['X_test'])
    accuracy = accuracy_score(dataset_handler.data['y_test'], y_pred)

    # Print results
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(dataset_handler.data['y_test'], y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset_handler.data['y_test'], y_pred))
    print(f"Cross-Validation Score: {cv_score:.2f}")

    # Get feature importance from the random forest
    feature_importance = pipeline.named_steps['rf'].feature_importances_
    print("Feature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Principal Component {i}: {importance:.4f}")

if __name__ == "__main__":
    main()