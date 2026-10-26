import torch
from torch.utils.data import Dataset, DataLoader 

class ImageFolderDataset(Dataset):
    def __init__(self,root,transform):
        super(ImageFolderDataset,self).__init__()
        self.root=Path(root)
        self.transform=transform
        self.paths=list(self.root.glob('*'))


    def __getitem__(self,idx):
        path=self.paths[idx]
        img=Image.open(str(path)).convert('RGB')
        img=self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def __name__(self):
        return 'ImageFolderDataset'