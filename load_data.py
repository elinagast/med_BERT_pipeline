import os
import pandas as pd
import csv

def read_dir(directory):
    data = pd.DataFrame()
    for path, _, files in os.walk(directory):
        for filename in files:
            temp = read_file(f'{path}/{filename}')
            data = pd.concat([data, temp], axis=0)
    return data

def read_file(path: str):
    if path.endswith('.xlsx'):
        data = pd.read_excel(path)
    elif path.endswith('.csv'):
        data = pd.read_csv(path)
    else:
        raise TypeError('Datatype is not supported. Only xlsx and csv are supported.')
    return data

def write_file(path: str, data):
    with open(path, 'x') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

