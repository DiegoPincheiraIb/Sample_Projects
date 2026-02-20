from src.dataset import IrisDatasetHandler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px


#! =============================================================================
N_ESTIMATORS      = 100 # Number of boosting rounds
LEARNING_RATE     = 0.1 # Step size shrinkage to prevent overfitting
MAX_DEPTH         = 3   # Maximum depth of a tree, increasing this value will make the model more complex and more likely to overfit
RANDOM_STATE      = 42  # Seed for reproducibility
EVAL_METRIC       = 'mlogloss' # Multiclass log loss, a common evaluation metric for multi-class classification problems
#! =============================================================================


def main():
    # Load and prepare the dataset
    dataset_handler = IrisDatasetHandler()
    num_samples, num_features, classes = dataset_handler.get_dataset_info()
    print(f"Dataset: {num_samples} samples, {num_features} features")
    print(f"Classes: {classes}")

    # Split the dataset into training and testing sets
    dataset_handler.split_dataset(test_size    = 0.2,
                                  random_state = 42)

    target_model = XGBClassifier(
            n_estimators      = N_ESTIMATORS,
            learning_rate     = LEARNING_RATE,
            max_depth         = MAX_DEPTH,
            random_state      = RANDOM_STATE,
            eval_metric       = EVAL_METRIC
        )

    # Train the model    
    target_model.fit(dataset_handler.data['X_train'],
                     dataset_handler.data['y_train'])
    
    # Evaluate the model
    y_pred = target_model.predict(dataset_handler.data['X_test'])
    accuracy = accuracy_score(dataset_handler.data['y_test'], y_pred)

    # Print results
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(dataset_handler.data['y_test'], y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset_handler.data['y_test'], y_pred))

    # Get feature importance
    feature_importance = target_model.feature_importances_
    print("Feature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i}: {importance:.4f}")

    # Visualizations with Plotly
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # Feature Importance Bar Chart
    fig_importance = go.Figure(data=[
        go.Bar(x=feature_names, y=feature_importance, marker_color='steelblue')
    ])
    fig_importance.update_layout(
        title='XGBoost Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        showlegend=False,
        height=500
    )
    fig_importance.write_html('feature_importance.html')
    print("\n✓ Feature importance plot saved to 'feature_importance.html'")

    # Confusion Matrix Heatmap
    cm = confusion_matrix(dataset_handler.data['y_test'], y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=dataset_handler.class_names,
        y=dataset_handler.class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500
    )
    fig_cm.write_html('confusion_matrix.html')
    print("✓ Confusion matrix plot saved to 'confusion_matrix.html'")

if __name__ == "__main__":
    main()