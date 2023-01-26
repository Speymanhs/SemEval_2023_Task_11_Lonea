import pandas as pd

path_to_csv = './resources/Dynamically_Generated_Hate_Dataset_v0.2.3.csv'

def get_dynamically_generated_data():
    df = pd.read_csv(path_to_csv)
    fin_list = []
    fin_labels = []
    for i in range(len(df)):
        fin_list.append(df['text'][i])
        label = df['label'][i]
        if label == 'hate':
            fin_labels.append(1.0)
        else:
            fin_labels.append(0.0)
    return fin_list, fin_labels

