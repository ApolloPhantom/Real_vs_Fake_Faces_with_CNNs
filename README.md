# Image Classification for Dataset of 10k Real vs. Fake Faces

This dataset was found on Kaggle([Link](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k)) and is a dataset containing 
10000 256x256 colour images of real and fake faces. There are 7000
images for training and 3000 images for validation.

## Objective

Our objective is to implement different popular Convolutional Neural
Network classifiers and show how much accuracy the classifiers can 
achieve using GPU-enabled Pytorch.


## Preprocessing

We load up our images from our data directory and load it into 
train and test dataloaders while transforming our images from 
256x256 to 64x64 , with a training batch size of 64. We downsample
our images to reduce training time.

![](outputs/output1.png)
![](outputs/output2.png)
![](outputs/output3.png)


## Training 

We train our models with the following loss function and optimization
function:-

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.0001)
```

O stands for Fake and 1 for Real.
### LeNet
<pre>
Accuracy on the test set: 72 %
Classification Report:
              precision    recall  f1-score   support

           0     0.6951    0.7887    0.7389      1500
           1     0.7558    0.6540    0.7012      1500

    accuracy                         0.7213      3000
   macro avg     0.7254    0.7213    0.7201      3000
weighted avg     0.7254    0.7213    0.7201      3000
</pre>
![](static/Plots/LeNet_Plot.png) 
![](static/Plots/LeNetcm.png)
![](static/Plots/LeNetGraph.png)
### AlexNet
<pre>
Accuracy on the test set: 76 %
Classification Report:
              precision    recall  f1-score   support

           0     0.8029    0.7007    0.7483      1500
           1     0.7345    0.8280    0.7784      1500

    accuracy                         0.7643      3000
   macro avg     0.7687    0.7643    0.7634      3000
weighted avg     0.7687    0.7643    0.7634      3000
</pre>
![](static/Plots/AlexPlot.png) 
![](static/Plots/Alexcm.png)
![](static/Plots/AlexNetGraph.png)
### Inception V1
<pre>
Accuracy on the test set: 75 %
Classification Report:
              precision    recall  f1-score   support

           0     0.7694    0.7140    0.7407      1500
           1     0.7332    0.7860    0.7587      1500

    accuracy                         0.7500      3000
   macro avg     0.7513    0.7500    0.7497      3000
weighted avg     0.7513    0.7500    0.7497      3000
</pre>
![](static/Plots/InceptionPlot.png) 
![](static/Plots/Inceptioncm.png)
![](static/Plots/InceptionGraph.png)
### VGG16
<pre>
Accuracy on the test set: 50 %
Classification Report:
              precision    recall  f1-score   support

           0     0.5000    1.0000    0.6667      1500
           1     0.0000    0.0000    0.0000      1500

    accuracy                         0.5000      3000
   macro avg     0.2500    0.5000    0.3333      3000
weighted avg     0.2500    0.5000    0.3333      3000
</pre>
![](static/Plots/VGGplot.png) 
![](static/Plots/VGGcm.png)
![](static/Plots/VGG16Graph.png)
### ResNet50
<pre>
Accuracy on the test set: 74 %
Classification Report:
              precision    recall  f1-score   support

           0     0.7031    0.8367    0.7641      1500
           1     0.7984    0.6467    0.7145      1500

    accuracy                         0.7417      3000
   macro avg     0.7507    0.7417    0.7393      3000
weighted avg     0.7507    0.7417    0.7393      3000
</pre>
![](static/Plots/Resnetplot.png) 
![](static/Plots/Resnetcm.png)
![](static/Plots/ResnetGraph.png)
### AlexNet (with Normalization every Convolution Layer)
<pre>
Accuracy on the test set: 64 %
Classification Report:
              precision    recall  f1-score   support

           0     0.5839    0.9787    0.7314      1500
           1     0.9342    0.3027    0.4572      1500

    accuracy                         0.6407      3000
   macro avg     0.7590    0.6407    0.5943      3000
weighted avg     0.7590    0.6407    0.5943      3000
</pre>
![](static/Plots/Alexv2Plot.png) 
![](static/Plots/Alexv2cm.png)
![](static/Plots/AlexNetv2Graph.png)
### VGG16 (with Normalization every Convolution Layer)
<pre>
Accuracy on the test set: 84 %
Classification Report:
              precision    recall  f1-score   support

           0     0.8455    0.8320    0.8387      1500
           1     0.8346    0.8480    0.8413      1500

    accuracy                         0.8400      3000
   macro avg     0.8401    0.8400    0.8400      3000
weighted avg     0.8401    0.8400    0.8400      3000
</pre>
![](static/Plots/VGG16v2plot.png) 
![](static/Plots/VGG16v2cm.png)
![](static/Plots/VGG16v2Graph.png)
### Custom CNN Architecture
<pre>
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
</pre>
![](Results/Accu_IntermediateCNN.png) 
![](Results/ConfusionMatrix_IntermediateCNN.png)
![](Results/Loss_IntermediateCNN.png)
## Conclusion

We can see that our altered VGG16 provides us the best result 
among all the predefined CNN model architectures we used for training and testing.The Custom CNN also provided excellent testing results but it simply began to overfit after 10 epochs. Further improvement can
be done by altering the structure of the best performing CNNs. We can also increase our training and validation images to get even better result.

Identifying Real and Fake faces have become quite challenging due the 
massive improvement in generating photorealistic images by AI. This project shows us how we can leverage different deep learning models to
differentiate between Real and Fake.


