#below here is the contrastive loss with SEResNext101
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SEResNext101

#this loss function is based off of this paper https://github.com/binh234/facial-liveness-detection/blob/main/train.ipynb
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):  #output1 is the first embedding #output2 is the second embedding #label for our case will always be 0 "Similar" becaause they are mere augmentations of each other
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True) #first compute the euclidian distance between the two
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2)) 
        #+ (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) #we only need this line of code if we had dissimilar embeddings but they hshould always be similar because they both have glaucoma or not
        return loss_contrastive


class ResNet(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=1, num_classes=1):
        super(ResNet, self).__init__()
        self.model = SEResNext101(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes
        ) 

        self.fc = nn.Linear(1024,1)
        self.relu = nn.ReLU()
     

    def forward(self, x1,x2):
        embedding1 = self.model(x1) #embedding1
        embedding2 = self.model(x2) #embedding2

        output = self.model(x1) #this is the output for binary classification
        output = self.relu(output)
        output = self.fc(output) #get 1
        
        return  embedding1,embedding2, torch.sigmoid(output)

# model = ResNet(spatial_dims=3, in_channels=1, num_classes=1024)
# input_tensor = torch.randn(1, 1, 64,64, 64)  
# input_tensor2 = torch.randn(1, 1, 64,64, 64)  

# emb1,emb2,output = model(input_tensor,input_tensor2)
# label = torch.Tensor([0])  
# contrastive_loss = ContrastiveLoss()
# loss = contrastive_loss(emb1, emb2, label)

# print(loss)
