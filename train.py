import argparse
from pathlib import Path
import torch
from utils.utils import ImageFolderDataset,get_transform 
from torch.utils.data import DataLoader
from models.

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\content_data',
                        help='Location of content dataset')
    
    parser.add_argument('--style_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\style_data',
                        help='location of style dataset')

    parser.add_argument('--vgg',type=str,default=r'C:\NST\Neural-Style-Transfer\vgg_normalised.pth',
                        help='location of pre-trained VGG')

    parser.add_argument('--experiment',type=str,default=r'experiment1',
                        help='Name of experiment')

    parser.add_argument('--final_size',type=int,default=256,
                        help='Size of Final Image')

    parser.add_argument('--content_size',type=int,default=512,
                        help='Size of Content Image')

    parser.add_argument('--style_size',type=int,default=512,
                        help='Size of Style Image')

    parser.add_argument('--crop',action='store_true',default=True,
                        help='Crop Image')

    parser.add_argument('--batch_size',type=int,default=4,
                        help='Batch Size')

    return parser.parse_args()
    

def main():
    args=parse_arguments()

    if torch.backends.mps.is_available():
        device=torch.device("mps")
    elif torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    save_dir=Path('experiments')/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)

    # Save arguments file
    with open(save_dir / 'args.text','w') as args_file:
        for key,value in vars(args).items():
            args_file.write(f'{key}:{value}\n')

    content_transform=get_transform(args.content_size,args.crop,args.final_size)
    style_transform=get_transform(args.style_size,args.crop,args.final_size)

    content_dataset=ImageFolderDataset(args.content_dir,content_transform)
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader=DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)

    style_dataloader=DataLoader(style_dataset,batch_size=args.batch_size,pin_memory=True,drop_last=True)

    print('Length -> content :',len(content_dataloader))
    print('Length -> style :',len(style_dataloader))













if __name__=='__main__':
    main()
