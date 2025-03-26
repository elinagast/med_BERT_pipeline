from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re

def plot_confusionmatrix(val_predictions, path: str, labels_names: list, mode):
    logits = val_predictions.predictions
    labels = np.array(val_predictions.label_ids)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()

    num_labels = labels.shape[1]

    # Einzelne Confusion Matrices für jede Klasse
    for i in range(num_labels):
        cm = confusion_matrix(labels[:, i], predictions[:, i])

        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.title(f"Confusion Matrix for {labels_names[i]} {mode}")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig(f'{path}/CM_label_{labels_names[i]}.png')
        plt.close()  # Speicher freigeben

    # Gesamte Co-Occurrence-Matrizen
    co_occurrence_1 = np.zeros((num_labels, num_labels))
    co_occurrence_0 = np.zeros((num_labels, num_labels))

    for i in range(labels.shape[0]):  
        positive_labels = labels[i] == 1
        negative_labels = labels[i] == 0

        co_occurrence_1 += np.outer(positive_labels, positive_labels)
        co_occurrence_0 += np.outer(negative_labels, negative_labels)

    # Positive Co-Occurrence Matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(co_occurrence_1, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    plt.title(f"Co-Occurrence Matrix for positive labels (1) {mode}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{path}/CM_total_true.png')
    plt.close()

    # Negative Co-Occurrence Matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(co_occurrence_0, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    plt.title(f"Co-Occurrence Matrix for negative labels (0) {mode}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.xticks(fontsize=10)  
    plt.yticks(fontsize=10)  
    plt.tight_layout()  
    plt.savefig(f'{path}/CM_total_false.png')
    plt.close()

def get_best_checkpoint(path):
    model_temp = ''
    checkpoints = []
    for checkpoint in os.listdir(path):
        if not os.path.isdir(os.path.join(path, checkpoint)):
            continue
        with open(os.path.join(path, checkpoint, 'trainer_state.json'), "r") as file:
            temp = json.load(file)
        if temp['best_model_checkpoint']:
            checkpoints.append({'path': re.findall('(checkpoint.+)', temp['best_model_checkpoint'])[0], 
                                'epochs': temp['epoch']})

    model_temp = max(checkpoints, key=lambda x:x['epochs'])
            
    with open(os.path.join(path, model_temp['path'], 'trainer_state.json'), "r") as best_checkpoint:
        return json.load(best_checkpoint)


def plot_training(path, save_path):
    # Load JSON file
    data = get_best_checkpoint(path)

    log_history = data["log_history"]  # Extract training logs

    metrics = {}  # Dictionary to store metrics over epochs

    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue  # Skip entries without an epoch

        for key, value in entry.items():
            if key not in ["epoch", "step"]:  # Skip non-metrics
                if key not in metrics:
                    metrics[key] = {"epochs": [], "values": []}
                metrics[key]["epochs"].append(epoch)
                metrics[key]["values"].append(value)

    # Plot each metric
    for metric, data in metrics.items():
        plt.figure()
        plt.plot(data["epochs"], data["values"], marker="o", linestyle="-", label=metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(f"Training Progress: {metric.replace('_', ' ').capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/training_{metric}.png")
        plt.close()

    plt.figure()
    plt.plot(metrics['eval_f1']["epochs"], metrics['eval_f1']["values"], marker="o", linestyle="-", label='Eval F1')
    plt.plot(metrics['eval_loss']["epochs"], metrics['eval_loss']["values"], marker="o", linestyle="-", label='Eval loss')
    plt.plot(metrics['eval_accuracy']["epochs"], metrics['eval_accuracy']["values"], marker="o", linestyle="-", label='Eval accuracy')
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.title(f"Training Progress: {metric.replace('_', ' ').capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/training_all_metrics.png")
    plt.close()