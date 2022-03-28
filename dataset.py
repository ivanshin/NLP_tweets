
import torch
import pandas as pd
from torch.utils.data import Dataset


class TweetsDataset(Dataset):
    def __init__(self,file_name):
        
        tweets_df=pd.read_csv(file_name, header= None)
        shape = tweets_df.shape


        x = tweets_df.iloc[0: shape[0], 1].values
        y = tweets_df.iloc[0: shape[0], 0].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
