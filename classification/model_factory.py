import monai 
import torch 
import torch.nn as nn
import torch.nn.functional as F

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

class ViTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ViTWrapper, self).__init__()
        self.vit = monai.networks.nets.ViT(*args, **kwargs)

    def forward(self, x):
        output, _ = self.vit(x)  # Get output and ignore hidden state
        return output

class ContrastiveWrapper(nn.Module):
    def __init__(self, base_model, contrastive_layer_size, num_classes):
        super(ContrastiveWrapper, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(contrastive_layer_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        embedding1 = self.base_model(x1)
        embedding2 = self.base_model(x2)
        
        output = self.base_model(x1)
        output = self.relu(output)
        output = self.fc(output)
        
        return embedding1, embedding2, output

def model_factory(model_name, dropout=.2, num_classes=2, contrastive_mode = "None", contrastive_layer_size = 512):
    n_classes = contrastive_layer_size if contrastive_mode != "None" else num_classes

    if model_name == "ViT":
        model = ViTWrapper(
            in_channels=1, 
            img_size=(128,512,64), 
            patch_size = (16,16,16),
            pos_embed='conv', 
            classification=True, 
            post_activation = "None",
            dropout_rate = dropout,
            num_classes = n_classes)
    elif model_name == "ResNext50":
        model = monai.networks.nets.SEResNext50(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout)
    elif model_name == "ResNext121":
        model = monai.networks.nets.SEResNext101(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout)
    else:
        raise ValueError(f"The model name {model_name} is not accepted.")
    

    if contrastive_mode in ["augmentation", "MacOp"]:
        return ContrastiveWrapper(model, contrastive_layer_size, num_classes)
    else:
        return model 
    