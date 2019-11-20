from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np

class TripletLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, start, end):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Images = pd.read_csv(csv_file)[start : end]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.Images
        Class = data.iloc[idx, 4]
        mode = data.iloc[idx, 5]

        X = []
        trainX1 = []
        trainX2 = []
        trainX3 = []
        id = idx
        while (len(X)==0):
            image_1 = cv2.imread(data.iloc[id, 1])
            image_2 = cv2.imread(data.iloc[id, 2])
            image_3 = cv2.imread(data.iloc[id, 3])
            if not (image_1 is None or image_2 is None or image_3 is None):
                if mode == 1:
                    trainX1.append(np.array(image_3))
                    trainX2.append(np.array(image_2))
                    trainX3.append(np.array(image_1))
                elif mode == 2:
                    trainX1.append(np.array(image_1))
                    trainX2.append(np.array(image_3))
                    trainX3.append(np.array(image_2))
                elif mode == 3:
                    trainX1.append(np.array(image_1))
                    trainX2.append(np.array(image_2))
                    trainX3.append(np.array(image_3))
                X.extend(trainX1)
                X.extend(trainX2)
                X.extend(trainX3)
            id += 1

        X = np.array(X).astype(np.float32).reshape(-1, 3, 224, 224)


        return X

def DATALoader(csv_file, val_ratio, num_workers, batch_size):
	data_len = len(pd.read_csv(csv_file))
	val_len = int(val_ratio*data_len)
	tr_dataset = TripletLoader(csv_file='data/labels.csv', start=0, end=data_len-val_len)
    val_dataset = TripletLoader(csv_file='data/labels.csv', start=[], end=-val_len)
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)