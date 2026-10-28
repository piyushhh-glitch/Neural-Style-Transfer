import os 
import torch
from flask import Flask,render_template,redirect,request,url_for,send_from_directory

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField,SubmitField,HiddenField,FloatField
from wtforms.validators import InputRequired,NumberRange
import torchvision
import torchvision.transforms as transforms
import io

from PIL import Image

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
    alpha=FloatField('Alpha',validators=[InputRequired(), NumberRange(min=0, max=1)])
    submit=SubmitField('Transfer Style')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder=VGGEncoder('vgg_normalised.pth').to(device)

decoder=Decoder().to(device)
weights=torch.load('decoder_final.pth',map_location=device,weights_only=True)

decoder.load_state_dict(weights,strict=True)

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



def style_transfer(content_image,style_image,encoder,decoder,alpha,device):
    content_transform=transforms.Compose(
        [
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ]
    )
    style_transforms=transforms.Compose(
        [
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ]
    )

    content_image=content_transform(content_image).unsqueeze(0).to(device)
    style_image=style_transforms(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_features=encoder(content_image)[-1]
        style_features=encoder(style_image)[-1]

        stylized_features=adaptive_instance_normalization(content_features,style_features)

        stylized_features=alpha * stylized_features + (1-alpha)*content_features

        stylized_images=decoder(stylized_features)

    return stylized_images



def save_image(image,path):
    image=image.detach().cpu().clone()
    image=image.squeeze(0)
    image=image.clamp(0,1)
    image=transforms.ToPILImage()(image)
    image.save(path)



@app.route('/',methods=['GET','POST'])
def index():
    form=UploadForm()
    result_image=None
    content_filename=None
    style_filename=None
    error=None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename=secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'],content_filename))
                form.content_path.data=os.path.join(app.config['UPLOAD_FOLDER'],content_filename)
        else:
            content_filename=form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename=secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'],style_filename))
                form.style_path.data=os.path.join(app.config['UPLOAD_FOLDER'],style_filename)
        else:
            style_filename=form.style_path.data

        if content_filename and style_filename:
            content_path=os.path.join(app.config['UPLOAD_FOLDER'],content_filename)
            style_path=os.path.join(app.config['UPLOAD_FOLDER'],style_filename)

            try:
                content_image=Image.open(content_path).convert("RGB")
                style_image=Image.open(style_path).convert("RGB")

                alpha=float(form.alpha.data)
                stylized_image=style_transfer(content_image,style_image,encoder,decoder,alpha,device)

                result_filename='stylized_'+content_filename
                result_path=os.path.join(app.config['UPLOAD_FOLDER'],result_filename)

                save_image(stylized_image,result_path)

                result_image=result_filename

                
            except Exception as e:
                error=str(e)

    elif request.method == 'POST':
        error = 'Please upload both content and style images.'

    return render_template('index.html',form=form,result_image=result_image,content_image=content_filename,style_image=style_filename,error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory("examples",filename)

if __name__=='__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost',5000,app,use_reloader=True,use_debugger=True)