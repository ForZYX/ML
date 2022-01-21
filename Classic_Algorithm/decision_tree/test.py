import pandas as pd
import numpy as np

if __name__ == '__main__':
    path = '../../../../../machine_learning/ML/Classic_Algorithm/data/CORK STOPPERS.XLS'
    data = pd.read_excel(path, sheet_name='Data').drop(0)
    print(data.head())