import argparse
from pathlib import Path
import torch

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default='C:\NST\Neural-Style-Transfer\content_data',
                        help='Location of content dataset')
    
    parser.add_argument('--style_dir',type=str,default='C:\NST\Neural-Style-Transfer\style_data',
                        help='location of style dataset')

    parser.add_argument('--vgg',type=str,default='',
                        help='location of pre-trained VGG')

    parser.add_argument('--experiment',type=str,default='experiment1',
                        help='Name of experiment')

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

if __name__=='__main__':
    main()
