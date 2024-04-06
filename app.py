from flask import render_template,jsonify,request,Flask
import os
import torchvision.transforms as transform
import torch
from torch import nn
from PIL import Image


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, 2))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# model = VGG16()
# model.load_state_dict(torch.load("models/VGG16.pth",map_location=torch.device('cpu')))
# model.eval()

model = torch.jit.load("models/VGG16.pth",map_location=torch.device('cpu'))
model.eval()

def transform_image(image_path):
    data_transform = transform.Compose([
    transform.Resize(size=(64,64)),
    transform.RandomHorizontalFlip(p=0.5),
    transform.ToTensor()
    ])
    image = Image.open(image_path)
    return data_transform(image)


def prediction(image_path):
    model.eval()
    with torch.no_grad():
        input_data = transform_image(image_path).unsqueeze(0)
        output = model.forward(input_data)
        _, y_hat = output.max(1)
    
    return int(y_hat)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/note")
def note():
    return render_template("note.html")


@app.get("/home")
def page():
    return render_template("predict.html")

@app.post('/upload')
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Error','filename': "No compatible file."})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file','filename': "No file selected."})
        
        if file and allowed_file(file.filename):
            if file.filename.rsplit('.', 1)[1].lower() != 'jpg':
                img = Image.open(file)
                file = os.path.splitext(file.filename)[0] + '.jpg'
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], file), 'JPEG')
            else:
                filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("File saved at:", filepath)
            res = prediction(filepath)
            os.remove(filepath)
            print(res)
            if res == 1:
                return jsonify({'filename': "REAL"})
            else:
                return jsonify({'filename': "FAKE"})
        else:
            return jsonify({'error': 'Error'})
    except Exception as e:
        print("Error:-",type(e).__name__)
        return jsonify({'error': 'Error','filename': "Error"})

if __name__ == '__main__':
    app.run(debug=True)
    



