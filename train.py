import torch 
import torch.nn as nn
from torchvision import transforms as t
from torch.utils.data import DataLoader

import sys


from dataset import MyDataset, plot_grid_images
from model import CNN_model


def evaluate(model, test_dataloader, criterion, device): ## This function will evaluate the model
    model.eval() ## put the model in evaluation model
    running_loss = 0.0
    accuracy = 0.0
    with torch.no_grad(): ## no need for gradients in evaluation
        for x, y in test_dataloader:
            x = x.to(device) ## send the data to device
            y = y.to(device)
            
            y_pred = model(x) ##forward pass
            loss = criterion(y_pred, y) ## calculate the loss
            
            running_loss += loss.item() ## calculate the running loss
            
            batch_accuracy = find_accuracy(y_pred, y)
            accuracy += batch_accuracy.item()
            
            
    loss = running_loss/len(test_dataloader) ## calculate the epoch loss
    epoch_accuracy = accuracy/len(test_dataloader)
    return loss, epoch_accuracy



def find_accuracy(y_pred, y_true):
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1) ## find the index of the maximum value
    correct = (y_pred == y_true).float() ##calculate the accuracy
    accuracy = correct.sum()/len(y_pred) ## calculate the accuracy
    return accuracy



## define function for calculating accuracy of the model with 2 classes using softmax function
# def find_accuracy(y_pred, y_true):
#     """
#     This function will calculate the accuracy of the model
#     """
#     y_pred = torch.softmax(y_pred, dim=1) ## apply softmax function
#     y_pred = torch.argmax(y_pred, dim=1) ## find the index of the maximum value
#     correct = (y_pred == y_true).float() ## calculate the accuracy
#     accuracy = correct.sum()/len(y_pred) ## calculate the accuracy
#     return accuracy ## return the accuracy



if __name__ == "__main__": ## execution da di zay na start kegii
    data_dir = 'Images'
    batch_size = 256
    input_size = 224 * 224 * 3
    hidden_size = 120 ###??????
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNN_model(in_channels=3, out_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    transforms = t.Compose([t.Resize((224, 224)), 
                        t.ToTensor()])

    ##Create the dataset
    train_dataset = MyDataset(data_dir, 'train', transform=transforms)
    test_dataset = MyDataset(data_dir, 'test', transform=transforms)

    ## create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ## Train the model
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        train_loss_per_epoch = 0
        train_accuracy_per_epoch = 0
        for i, (images, labels) in enumerate(train_dataloader):
            ##Move tensor to the configured device
            # images = images.view(batch_size, channels, height, width)
            # images = images.reshape(-1,224 * 224 * 3).to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            ##forward pass --> model predictions
            outputs = model(images)
            
            ## loss calculations 
            loss = criterion(outputs, labels)
            
            ## backward and optimize
            optimizer.zero_grad()
            loss.backward() ## gradient calculation for each weight in the model using chain-rule
            optimizer.step() # actual updates of the model weights
            
            
            # # if (i+1) % 1 == 0:
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}'
            #             .format(epoch+1, num_epochs, i+1, total_step, loss.item())) ## show epoch and batch size progress
                
            Accuracy = find_accuracy(outputs, labels)
                
            # print('Accuracy of the network on the train images: {} %'.format(Accuracy))
            
            train_accuracy_per_epoch += Accuracy.detach().cpu()
            train_loss_per_epoch += loss.detach().cpu()

        average_train_accuracy = train_accuracy_per_epoch / len(train_dataloader)
        average_train_loss = train_accuracy_per_epoch / len(train_dataloader)
            
            
            
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
        
        
        print(f"Epoch {epoch}'s Results: ")
        print(f"Train Loss: {average_train_loss}, Test Loss: {test_loss}")
        print(f"Train Accuracy: {average_train_accuracy}, Test Accuracy: {test_accuracy}")
        print("**************\n")
        
        ## print every 10 epochs
        # if i % 10 == 0:
        #     print(f'Epoch: {i} and loss: {loss}')
        
        
        
        

            


                
                
                

            
            

            
            
            


                            

