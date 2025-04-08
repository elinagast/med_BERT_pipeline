from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from models import MultiLabelDataset
from transformers.trainer_utils import PredictionOutput
import numpy as np

def train_model(train_texts, val_texts, train_labels, val_labels, model_name, dest_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Texte tokenisieren
    train_encodings = tokenizer(train_texts, padding='max_length', truncation=True)
    val_encodings = tokenizer(val_texts, padding='max_length', truncation=True)
    
    # Train- und Validation-Datasets erstellen
    train_dataset = MultiLabelDataset(train_encodings, train_labels)
    val_dataset = MultiLabelDataset(val_encodings, val_labels)

    # Modell laden
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(val_labels[0]),  # Anzahl der Labels (z. B. Fieber, Husten, Müdigkeit)
        problem_type="multi_label_classification"
    ).to(device)

    # Trainingsparameter
    training_args = TrainingArguments(
        output_dir=f"{dest_path}/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=400,
        logging_dir="./logs",
        load_best_model_at_end=True,
        save_total_limit=500,
        logging_steps=10
    )
    
    # Trainer initialisieren
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    ), tokenizer

# Metrics
def compute_metrics(eval_pred):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits, device=device)) > 0.5  # Schwellenwert für Multi-Label

    predictions = predictions.cpu().numpy()

    # Initialisierung einer Liste, um Confusion Matrices für jedes Label zu speichern
    confusion_matrices = []
    num_labels = labels.shape[1]

    for i in range(num_labels):
        cm = confusion_matrix(labels[:, i], predictions[:, i])  # Confusion Matrix für jedes Label
        confusion_matrices.append(cm)
        print(f"Confusion Matrix for Label {i}:\n{cm}\n")

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

def predict_model(model_name, texts, true_labels):

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Logits are the raw prediction values
        predictions = torch.sigmoid(logits)  # Apply sigmoid if you're predicting binary labels
        predicted_labels = (predictions > 0.5).int()

    val_predictions = PredictionOutput(predictions=logits.numpy(), label_ids=np.array(true_labels))    
    return val_predictions
