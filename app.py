import os 
import torch
from flask import Flask,render_template,redirect,request,url_for,send_from_directory

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField,SubmitField,HiddenField,FloatField
from wtforms.validators import InputRequired
import torchvision
import torchvision.transforms as transfroms
import io

from utils.model import VGGEncoder,Decoder
from utils.utils import calc_mean_std,adaptive_instance_normalization

app=Flask(__name__)

app.config['SECRET_KEY']='nstsecretkey'
app.config['UPLOAD_FOLDER']='static/uploads'
app.config['ALLOWED_EXTENSIONS']={'png','jpg','jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)

class UploadForm(FlaskForm):
    content=FileField('Content Image')
    style=FileField('Style Image')
    content_path=HiddenField()
    style_path=HiddenField()
    alpha=FloatField('Alpha',default=1.0)
    submit=SubmitField('Transfer Style')

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder=VGGEncoder('vgg_normalised.pth').to(device)

    decoder=Decoder().to(device)
    decoder.load_state_dict(torch.load('decoder_final.pth'))

    encoder.eval()
    decoder.eval()
