import cv2
from torch.utils.data import DataLoader, Dataset
import numpy as np



class FracAtlasDataset(Dataset):

    def __init__(self, dataframe, mode = "train", transforms = None):
        
        super().__init__()
        self.image_ids = dataframe['filename'].unique()
        self.df = dataframe
        # self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index: int):

        
        image_id = self.image_ids[index]
        records = self.df[self.df['filename'] == image_id]
        

  
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

       
        if self.mode == "train" or self.mode == "valid":

            # Converting xmin, ymin, w, h to x1, y1, x2, y2
            boxes = np.zeros((records.shape[0], 5))
            boxes[:, 0:4] = records[['xmin', 'ymin', 'xmax', 'ymax']].values
            
            boxes[:, 4] = 1 if records['class'].values.any() == "fractured" else 0
            
            
            sample = {'img': image, 'annot': boxes}
                
            if self.transforms:
                sample = self.transforms(sample)

            return sample
        

    def __len__(self) -> int:
        return self.image_ids.shape[0]