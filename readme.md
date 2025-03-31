# German Med-BERT Training and Prediction

This repository provides a pipeline for training and evaluating a German Med-BERT model for medical text classification. The script supports both training and prediction modes.

## 🚀 Features
	•	Preprocesses medical text data for training and inference.
	•	Trains a multi-label classification model using GerMedBERT.
	•	Evaluates model performance and generates plots (confusion matrices, training progress).
	•	Supports inference on new medical text data.

## Installation
### Clone Repository
```
git clone https://github.com/your-repo-name.git
cd your-repo-name
```
### Install Dependencies
```
pip install -r requirements.txt
```

## 🏗 Usage

The script supports two modes:
	•	Training (train)
	•	Prediction (predict)

🔹 1️⃣ Training Mode

Train the German Med-BERT model using labeled medical text data.
```
python main.py --mode train --data path/to/data.csv --rp results_dir --t text_column_name
```

|Argument|Description|
|--mode|Takes train or predict. Runs the training or prediction process.|
|--data|Path to the dataset or directory for inference.|
|--rp|Directory where results will be saved.|
|--t|Name of the text column(s) (Must be the same for all input datasets).|

## 📊 Output
	•	Trained Model (results_dir/results/)
	•	Evaluation Metrics (F1-score, accuracy, loss)
	•	Plots:
	•	Confusion Matrix (training_confusion_matrix.png)
	•	Training Progress (training_loss.png, training_f1.png, training_accuracy.png)

## 🛠 Project Structure

```
/project-root
│── main.py                   # Main script for training and inference
│── preprocessing.py           # Data cleaning and preprocessing
│── model_prep.py              # Model preparation and training functions
│── load_data.py               # Data loading utilities
│── models.py                  # Custom dataset classes
│── plot_results.py            # Plot generation (confusion matrix, training metrics)
│── requirements.txt           # Required dependencies
│── results/                   # Directory for model outputs and plots
```

## 📢 Acknowledgment

This project leverages GerMedBERT, a pretrained transformer model for medical NLP in German. Model available on Hugging Face.

For more details, visit GerMedBERT on Hugging Face.
