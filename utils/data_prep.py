from torch.utils.data import Dataset, DataLoader
import cv2, os
import torch
import numpy as np
import pandas as pd

class TripletLoader(Dataset):
    """Google face comparing dataset."""

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

        X = []
        trainX1 = []
        trainX2 = []
        trainX3 = []
        id = idx
        while (len(X)==0):
            if id >= len(self.Images):
                id = 0
            Class = self.Images.iloc[id, 3]
            mode = self.Images.iloc[id, 4]

            if not os.path.exists(self.Images.iloc[id, 0]):
                id = id + 1
                continue
            else:
                image_1 = cv2.imread(self.Images.iloc[id, 0])
                if image_1.shape != (224, 224, 3):
                    id = id + 1
                    continue
            if not os.path.exists(self.Images.iloc[id, 1]):
                id = id + 1
                continue
            else:
                image_2 = cv2.imread(self.Images.iloc[id, 1])
                if image_2.shape != (224, 224, 3):
                    id = id + 1
                    continue
            if not os.path.exists(self.Images.iloc[id, 2]):
                id = id + 1
                continue
            else:
                image_3 = cv2.imread(self.Images.iloc[id, 2])
                if image_3.shape != (224, 224, 3):
                    id = id + 1
                    continue
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

        # Class = self.Images.iloc[idx, 3]
        # mode = self.Images.iloc[idx, 4]
        #
        # image_1 = cv2.imread(self.Images.iloc[id, 0])
        # image_2 = cv2.imread(self.Images.iloc[id, 1])
        # image_3 = cv2.imread(self.Images.iloc[id, 2])
        # if mode == 1:
        #     trainX1.append(np.array(image_3))
        #     trainX2.append(np.array(image_2))
        #     trainX3.append(np.array(image_1))
        # elif mode == 2:
        #     trainX1.append(np.array(image_1))
        #     trainX2.append(np.array(image_3))
        #     trainX3.append(np.array(image_2))
        # elif mode == 3:
        #     trainX1.append(np.array(image_1))
        #     trainX2.append(np.array(image_2))
        #     trainX3.append(np.array(image_3))
        # X.extend(trainX1)
        # X.extend(trainX2)
        # X.extend(trainX3)

        X = np.array(X).astype(np.float32).reshape(-1, 3, 224, 224)


        return X

def DATALoader(csv_file, args):
    data_len = len(pd.read_csv(csv_file))
    val_len = int(args.val_ratio * data_len)
    tr_dataset = TripletLoader(csv_file=csv_file, start=0, end=int((data_len - val_len)*args.tr_ratio))
    val_dataset = TripletLoader(csv_file=csv_file, start=-val_len, end=None)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
    return  tr_dataloader, val_dataloader