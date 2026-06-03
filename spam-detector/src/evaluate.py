import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, plot_dir="outputs/plots"):
    """Computes F1/Accuracy scores and exports heatmaps."""
    os.makedirs(plot_dir, exist_ok=True)

    target_names = ['Ham (Normal)', 'Spam']
    y_pred = model.predict(X_test)

    print("\n=========================================")
    print("        MODEL EVALUATION RESULTS         ")
    print("=========================================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Spam Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "spam_confusion_matrix.png"))
    plt.close()
