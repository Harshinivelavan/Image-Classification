# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.
Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.
Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.

## Neural Network Model

<img width="1027" height="441" alt="image" src="https://github.com/user-attachments/assets/f45a9772-ba3e-4215-9f01-cdcedeba61dd" />


## DESIGN STEPS

STEP 1:


Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:


Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 3:


Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 4:


Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 5:


Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 5:


Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name:V.Harshini
### Register Number:212224040109
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: V.Harshini')
        print('Register Number: 212224040109')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')



```

## OUTPUT
### Training Loss per Epoch

<img width="401" height="262" alt="image" src="https://github.com/user-attachments/assets/82a0dd74-168f-4c10-a94a-35b89e013413" />


### Confusion Matrix

<img width="884" height="746" alt="image" src="https://github.com/user-attachments/assets/f886a4e3-dbd6-4784-86c6-817c71c4163f" />


### Classification Report

<img width="592" height="372" alt="image" src="https://github.com/user-attachments/assets/5ed7b352-0014-4eac-922c-669e3bd00dda" />



### New Sample Data Prediction

<img width="495" height="568" alt="image" src="https://github.com/user-attachments/assets/f3985783-1924-4183-86e0-a3f47d83a675" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
