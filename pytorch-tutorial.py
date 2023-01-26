import torch
from torch.utils.data import Dataset




import json
def prepare_dataset(dataset_list, splits):
    path_to_extracted_folder = './resources/data_practicephase_cleardev/'
    train_ds = []
    dev_ds = []
    for current_dataset in dataset_list:                         # loop on datasets
        for current_split in splits:                                                                 # loop on splits, here only train
            current_file = path_to_extracted_folder+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
            if current_split == 'train':
                train_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
            else:
                dev_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
    return train_ds, dev_ds

train_ds, dev_ds = prepare_dataset(['HS-Brexit'], ['train', 'dev'])

train_df = pd.DataFrame.from_dict(train_ds[0], orient="index")
dev_df = pd.DataFrame.from_dict(dev_ds[0], orient="index")

train_hard_labels = [float(soft_label) for soft_label in train_df['hard_label']]
dev_hard_labels = [float(soft_label) for soft_label in dev_df['hard_label']]

tf_train_text = train_df['text']
tf_train_list = [text for text in tf_train_text]
tf_dev_text = dev_df['text']
tf_dev_list = [text for text in tf_dev_text]

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X =
        self.y = ...

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]