import argparse
import os
import preprocessing
import load_data
import model_prep
import plot_results
import models
import numpy as np

parser = argparse.ArgumentParser('Training and Predicting for a German Med-Bert Algorithm')
parser.add_argument('--rp', action='store', dest='results_path', type=str)
parser.add_argument('--data', action='store', dest='data', type=str)
parser.add_argument('-model_name', action='store', dest='model_name', type=str, default='GerMedBERT/medbert-512')
parser.add_argument('--t', action='store', dest='texts', nargs='+', default=[])
parser.add_argument('--mode', dest='mode')
#parser.add_argument('--metrics', dest='metrics')

def preprocess_training(data):
    data = preprocessing.clean_data(data)
    texts = preprocessing.extract_text(data, args.texts)
    labels = preprocessing.extract_labels(data, args.texts)
    return preprocessing.split_data(texts, labels)

def preprocess_predict(data):
    data = preprocessing.clean_data(data)
    texts = preprocessing.extract_text(data, args.texts)
    labels = preprocessing.extract_labels(data, args.texts)
    return texts, labels


args = parser.parse_args()
if args.data:
    if not os.path.exists(args.data):
        raise Exception('Path does not exists.')
    if os.path.isdir(args.data):
        data = load_data.read_dir(args.data)
    if os.path.isfile(args.data):
        data = load_data.read_file(args.data)
    

if args.mode == 'train':
    train_texts, val_texts, train_labels, val_labels = preprocess_training(data)
    trainer, tokenizer = model_prep.train_model(train_texts, val_texts, train_labels, val_labels, args.model_name, args.results_path)
    
    #define validation dataset
    val_encodings = tokenizer(val_texts, padding="max_length", truncation=True, max_length=512)
    val_dataset = models.MultiLabelDataset(val_encodings, val_labels)

    # Training starten
    trainer.train()

    trainer.save_model(rf"{args.results_path}/results")
    tokenizer.save_pretrained(rf"{args.results_path}/results")
    results = trainer.evaluate()

    predictions = trainer.predict(val_dataset)

    plot_results.plot_confusionmatrix(predictions, f"{args.results_path}/results", data.drop(args.texts, axis=1).columns, args.mode)
    plot_results.plot_training(f"{args.results_path}/results", f"{args.results_path}/results")

if args.mode == 'predict':
    texts, labels = preprocess_predict(data)

    predictions = model_prep.predict_model(args.model_name, texts, labels)
    plot_results.plot_confusionmatrix(predictions, f"{args.results_path}/results", data.drop(args.texts, axis=1).columns, args.mode)

    
    