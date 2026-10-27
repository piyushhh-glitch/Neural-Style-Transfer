import argparse
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from torch.utils.data import DataLoader
from utils.model import VGGEncoder,Decoder
import torch.nn as nn 
from torchvision.utils import save_image

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\content_data',
                        help='Location of content dataset')
    
    parser.add_argument('--style_dir',type=str,default=r'C:\NST\Neural-Style-Transfer\style_data',
                        help='location of style dataset')

    parser.add_argument('--vgg',type=str,default=r'C:\NST\Neural-Style-Transfer\vgg_normalised.pth',
                        help='location of pre-trained VGG')

    parser.add_argument('--experiment',type=str,default='experiment1',
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

    parser.add_argument('--lr',type=float,default=1e-4,
                        help='Learning Rate')

    parser.add_argument('--lr_decay',type=float,default=5e-5,
                        help='Learning Rate Decay')

    parser.add_argument('--epochs',type=int,default=2,
                        help='Number of epochs')

    parser.add_argument('--content_weight',type=float,default=1.0,
                        help='Content Weight')

    parser.add_argument('--style_weight',type=float,default=5,
                        help='Style Weight')

    parser.add_argument('--log_interval',type=int,default=1,
                        help='Log Interval')

    parser.add_argument('--save_interval',type=int,default=2,
                        help='Save Interval')

    parser.add_argument('--resume',action='store_true',default=False,
                        help='Resume Training')

    parser.add_argument('--decoder_path',type=str,default=None,
                        help='Path to decoder checkpoint')

    parser.add_argument('--optimizer_path',type=str,default=None,
                        help='Path to optimizer checkpoint')
    
    return parser.parse_args()
    

def main():
    args=parse_arguments()

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')


    save_dir=Path('experiments')/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)

    # Save arguments file
    with open(save_dir / 'args.txt','w') as args_file:
        for key,value in vars(args).items():
            args_file.write(f'{key}:{value}\n')

    content_transform=get_transform(args.content_size,args.crop,args.final_size)
    style_transform=get_transform(args.style_size,args.crop,args.final_size)

    content_dataset=ImageFolderDataset(args.content_dir,content_transform)
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader=DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=(device.type=='cuda'),drop_last=True)

    style_dataloader=DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=(device.type=='cuda'),drop_last=True)

    print('Length -> content :',len(content_dataloader))
    print('Length -> style :',len(style_dataloader))

    encoder=VGGEncoder(args.vgg).to(device)

    encoder.eval()

    decoder=Decoder().to(device)

    decoder_path=Path('decoder_final.pth')

    weights=torch.load(
        decoder_path,
        map_location=device,
        weights_only=True
    )

    weights = {
        key.replace("net.", "decoder.", 1)
        if key.startswith("net.") else key: value
        for key, value in weights.items()
    }

    decoder.load_state_dict(weights, strict=True)

    print("Pretrained AdaIN decoder loaded successfully!")


    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)

    scheduler=optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch:1.0/(1.0+args.lr_decay*epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path,map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path,map_location=device))

    mse_loss=nn.MSELoss()

    encoder.eval()
    running_loss=None
    running_content_loss=None
    running_style_loss=None

    for epoch in range(args.epochs):

        progress_bar=tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))

        running_loss=0.0
        running_content_loss=0.0
        running_style_loss=0.0

        for content_batch,style_batch in progress_bar:

            content_batch=content_batch.to(device)
            style_batch=style_batch.to(device)

            content_features=encoder(content_batch)
            style_features=encoder(style_batch)

            t=adaptive_instance_normalization(content_features[-1],style_features[-1])

            generated_images=decoder(t)
            generated_features=encoder(generated_images)

            content_loss=mse_loss(generated_features[-1],t)*args.content_weight

            style_loss=0

            for g_f,s_f in zip(generated_features,style_features):

                g_mean,g_std=calc_mean_std(g_f)
                s_mean,s_std=calc_mean_std(s_f)

                style_loss+=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std)

            style_loss=style_loss*args.style_weight

            loss=content_loss+style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch:{epoch+1},Loss:{loss.item():4f},Content Loss:{content_loss.item():4f},Style Loss:{style_loss.item():4f}')

            running_loss+=loss.item()
            running_content_loss+=content_loss.item()
            running_style_loss+=style_loss.item()

        scheduler.step()

        running_loss/=len(content_dataloader)
        running_content_loss/=len(content_dataloader)
        running_style_loss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iteration {epoch+1}/{args.epochs} , Loss => {running_loss:4f}, Content loss => {running_content_loss:4f}, Style loss=> {running_style_loss:4f}')


        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(),save_dir/f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(),save_dir/f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output=torch.cat([content_batch,style_batch,generated_images],dim=0)
                save_image(output,save_dir/f'output_{epoch+1}.png',nrow=args.batch_size)


if __name__=='__main__':
    main()
