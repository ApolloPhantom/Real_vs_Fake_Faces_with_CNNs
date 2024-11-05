from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

class IntermediateCNN(nn.Module):
    def __init__(self, num_classes):
        super(IntermediateCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = IntermediateCNN(num_classes=2)
model.load_state_dict(torch.load('Models/model.pth', map_location=torch.device('cpu')))
model.eval()

h, w = 64, 64
data_transform = transforms.Compose([
    transforms.Resize(size=(h, w)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return data_transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        with torch.no_grad():
            outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_name = 'Fake' if predicted.item() == 0 else 'Real'
        return jsonify({'result': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
