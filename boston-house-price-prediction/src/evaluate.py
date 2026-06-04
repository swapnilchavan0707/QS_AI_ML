import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(test_path="data/processed/test.csv", model_path="outputs/models/linear_regression_model.pkl",
                   plot_dir="outputs/plots"):
    """Evaluates the saved model against testing data and exports a performance plot."""
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(test_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Required test data or saved model file is missing.")

    print("Loading test dataset and trained model...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=['MEDV'])
    y_test = test_df['MEDV']

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n======================= MODEL EVALUATION METRICS =======================")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared Accuracy Score: {r2:.4f}")
    print("========================================================================")

    # Generate and save actual vs predicted plot
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6, color='purple', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MEDV (Price in $1000s)')
    plt.ylabel('Predicted MEDV (Price in $1000s)')
    plt.title('Actual vs. Predicted Boston House Prices')
    plt.grid(True)

    plot_path = os.path.join(plot_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Success! Performance scatter plot exported to: {plot_path}")


if __name__ == "__main__":
    evaluate_model()
