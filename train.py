import argparse
from pathlib import Path
import torch
from utils.utils import ImageFolderDataset

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\content_data',
                        help='Location of content dataset')
    
    parser.add_argument('--style_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\style_data',
                        help='location of style dataset')

    parser.add_argument('--vgg',type=str,default=r'',
                        help='location of pre-trained VGG')

    parser.add_argument('--experiment',type=str,default=r'experiment1',
                        help='Name of experiment')

    parser.add_argument('--final_size',type=int,default=512,
                        help='Size of Final Image')

    parser.add_argument('--content_size',type=int,default=256,
                        help='Size of Content Image')

    parser.add_argument('--style_size',type=int,default=256,
                        help='Size of Style Image')

    parser.add_argument('--crop',action='store_true',default=True
                        help='Crop Image')

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



if __name__=='__main__':
    main()
