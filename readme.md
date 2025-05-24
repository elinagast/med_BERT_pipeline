# [GerMedBERT](https://huggingface.co/GerMedBERT/medbert-512) Training and Prediction

This repository provides a pipeline for training and evaluating a German Med-BERT model for medical text classification. The script supports both training and prediction modes.

## 🚀 Features
- Preprocesses medical text data for training and inference.
- Trains a multi-label classification model using [GerMedBERT](https://huggingface.co/GerMedBERT/medbert-512).
- Evaluates model performance and generates plots (confusion matrices, trainingprogress).
- Supports inference on new medical text data.

## Installation
### Clone Repository
```
git clone https://github.com/elinagast/med_BERT_pipeline.git
```
### Install Dependencies
```
pip install -r requirements.txt
```

## 🏗 Usage

The script supports two modes:
	•	Training (train)
	•	Prediction (predict)


🔹 **Training** Mode
Train the German Med-BERT model using labeled medical text data.
```
python main.py --mode train --data path/to/data.csv --rp results_dir --t text_column_name
```

🔹 **Prediction** Mode
Use the trained model to make predictions on new data.
```
python main.py --mode predict --data path/to/data.csv --rp results_dir --t text_column_name -model_name path/to/results_dir
```

|Argument|Description|Input type|
|:-----|:-----|:-----|
|--mode|Takes train or predict. Runs the training or prediction process.|train/predict|
|--data|Path to the dataset or directory for inference. Allowed Data types: csv, excel|Path|
|--rp|Directory where results will be saved.|Path|
|--t|Name of the text column(s) (Must be the same for all input datasets).|String/String list|
|-model_name|Name of the text column(s) (Must be the same for all input datasets). Uses [GerMedBERT](https://huggingface.co/GerMedBERT/medbert-512) as default. |Path/None|

## 📊 Output
- Finetuned Model (results_dir/results/)
- Evaluation Metrics (F1-score, accuracy, loss)
- Plots:
    - Confusion Matrix (training_confusion_matrix.png)
    - Training Progress (training_loss.png, training_f1.png, training_accuracy.png)

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

## Inference
Download a pretrained model [here](https://mega.nz/file/iZlE2YhI#6TOFhoL9m8E5m-HF4U9jwL-lBjIyu8a3k_pzyhc3Dhc).
This model predicts on the following diseases:
- SchilddrüsenläsioStruma
- Lungenmetastasen
- Lungenoduli_Granulome_(nicht_suspekt)
- Arteriosklerose
- Pleuraerguss
- Perikarderguss
- Pneumonie
- Leberzysten
- Lebermetastasen
- Gallensteine
- Nierenzysten
- Aszites_Freie_flüssigkeit
- Lymphadenopathie
- Maligner_tumor
- Entzündlicher_prozess
- Nebennierenraumforderung
- Knochenmetastase
- Degenerative_Skelettveränderungen


## 📢 Acknowledgment

This project leverages [GerMedBERT](https://huggingface.co/GerMedBERT/medbert-512), a pretrained transformer model for medical NLP in German. Model available on Hugging Face.

For more details, visit [GerMedBERT](https://huggingface.co/GerMedBERT/medbert-512) on Hugging Face.
