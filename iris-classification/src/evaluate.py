import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_and_plot(model, X_test, y_test, df_original, plot_dir="outputs/plots"):
    """Evaluates model performance and saves visual matrix plots."""
    os.makedirs(plot_dir, exist_ok=True)

    target_names = ['Setosa', 'Versicolor', 'Virginica']
    y_pred = model.predict(X_test)

    # Generate terminal metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=========================================")
    print("        MODEL EVALUATION RESULTS         ")
    print("=========================================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Create Confusion Matrix Plot
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Flower Species")
    plt.ylabel("True Flower Species")
    plt.tight_layout()

    # Save chart to outputs folder
    plot_path = os.path.join(plot_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Performance plot saved to: {plot_path}")
