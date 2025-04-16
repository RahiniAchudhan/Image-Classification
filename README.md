# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Design and implement a Convolutional Neural Network (CNN) to classify grayscale images from the FashionMNIST dataset into 10 distinct categories. The model should learn to recognize patterns and features in the images to accurately predict their respective classes.

## Dataset
![Screenshot 2025-04-16 132100](https://github.com/user-attachments/assets/978aece5-7113-4e9a-afb4-6d9a34c076a3)

## Neural Network Model

![Screenshot 2025-03-24 120704](https://github.com/user-attachments/assets/052ed718-d46b-4489-b0d7-af85effed8e6)


## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name:
### Register Number:
```
class CNNClassifier(nn.Module):
  def __init__(self): # Define __init__ method explicitly
    super(CNNClassifier, self).__init__() # Call super().__init__() within __init__
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Correct argument names
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Correct argument names
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Correct argument names
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjust input size for Linear layer (Calculation needs update if image size changed)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x))) # Correctly call self.conv1
    x = self.pool(torch.relu(self.conv2(x)))  # Correctly call self.conv2
    x = self.pool(torch.relu(self.conv3(x))) # Correctly call self.conv3
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = torch.relu(self.fc1(x)) # Correctly call self.fc1
    x = torch.relu(self.fc2(x)) # Correctly call self.fc2
    x = self.fc3(x)
    return x



```

```
# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)
```

```
# Train the Model
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    print('Name:RAHINI A')
    print('Register Number:212223230165')

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

        # Print only once per epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

```

## OUTPUT
### Training Loss per Epoch
![Screenshot 2025-03-28 054057](https://github.com/user-attachments/assets/5c486d8d-e6be-4bc9-8930-1455fcd04e32)


### Confusion Matrix
![Screenshot 2025-04-07 100835](https://github.com/user-attachments/assets/b0cd31c6-3c29-4603-965c-fd5229c32ed8)


### Classification Report
![Screenshot 2025-04-07 100932](https://github.com/user-attachments/assets/b9ea4201-f2f6-460d-bf0e-be1a990b58ce)


### New Sample Data Prediction
![Screenshot 2025-04-07 100957](https://github.com/user-attachments/assets/c1133369-0a78-48a4-9698-58609d6ec6ae)

## RESULT
The CNN model was successfully trained on the MNIST dataset for 3 epochs, achieving a final loss of 0.1632, demonstrating effective handwritten digit classification.
