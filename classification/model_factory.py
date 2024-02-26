import monai 
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from classification.models import * 

# class MedicalNet(nn.Module):

#   def __init__(self, path_to_weights, device, dropout_prob):
#     super(MedicalNet, self).__init__()
#     self.model = resnet200(sample_input_D=1024, sample_input_H=200, sample_input_W=200, num_seg_classes=2)
#     self.model.conv_seg = nn.Sequential(
#         nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
#         nn.Flatten(start_dim=1),
#         nn.Dropout(dropout_prob)
#     )
    # net_dict = self.model.state_dict()
    # pretrained_weights = torch.load(path_to_weights, map_location=torch.device("cuda:0"))
    # pretrain_dict = {
    #     k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
    #   }
    # net_dict.update(pretrain_dict)
    # self.model.load_state_dict(net_dict)
    # self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 2))
    # self.fc = nn.Linear(2048, 2)

#   def forward(self, x):
#     features = self.model(x)
#     return self.fc(features)

    
class Efficient3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, conv_layers=[32,64], fc_layers=[], dropout_rate=0.5):
        super(Efficient3DCNN, self).__init__()

        # Conv3D layers
        layers = [nn.Conv3d(in_channels if i == 0 else conv_layers[i-1], 
                            conv_layers[i], kernel_size=3, stride=2, padding=1) 
                  for i in range(len(conv_layers))]
        self.conv_layers = nn.Sequential(*layers, nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)  # Correct placement

        # FC layers
        in_features = conv_layers[-1]
        self.fc_layers = nn.Sequential()
        for out_features in fc_layers:
            self.fc_layers.add_module('fc', nn.Linear(in_features, out_features))
            self.fc_layers.add_module('dropout', nn.Dropout(dropout_rate))
            in_features = out_features
        self.fc_layers.add_module('fc_final', nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling here
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



#this loss function is based off of this paper https://github.com/binh234/facial-liveness-detection/blob/main/train.ipynb
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):  #output1 is the first embedding #output2 is the second embedding #label for our case will always be 0 "Similar" becaause they are mere augmentations of each other
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True) #first compute the euclidian distance between the two
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2)) 
        #+ (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) #we only need this line of code if we had dissimilar embeddings but they hshould always be similar because they both have glaucoma or not
        return loss_contrastive

class ViTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ViTWrapper, self).__init__()
        self.vit = monai.networks.nets.ViT(*args, **kwargs)

    def forward(self, x):
        output, _ = self.vit(x)  # Get output and ignore hidden state
        return output
    
class ResNetWrapper(nn.Module):
    def __init__(self, name, dropout_prob=.2, *args, **kwargs):
        super(ResNetWrapper, self).__init__()
        if name == "ResNet10":
            self.resnet = monai.networks.nets.resnet10(*args, **kwargs)
        else:
            self.resnet = monai.networks.nets.resnet18(*args, **kwargs)
        # self.fc1 = nn.Linear(512, 64)
        # self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(512, 2)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        output = self.resnet(x)
        output = self.fc3(self.dropout(output))

        # output = self.fc2(self.relu(self.fc1(output)))
        return output
    
class ContrastiveWrapper(nn.Module):
    def __init__(self, base_model, contrastive_layer_size, num_classes, dropout_rate, join_method='concat'):
        super(ContrastiveWrapper, self).__init__()
        self.base_model = base_model
        self.join_method = join_method
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Adjust the input size of the fully connected layer based on the joining method
        if join_method == 'concat':
            fc_input_size = 2 * contrastive_layer_size
        elif join_method == 'sum':
            fc_input_size = contrastive_layer_size
        else:
            raise ValueError("join_method must be either 'concat' or 'sum'")

        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x1, x2):
        embedding1 = self.base_model(x1)
        embedding2 = self.base_model(x2)

        # Joining the embeddings
        if self.join_method == 'concat':
            combined = torch.cat((embedding1, embedding2), dim=1)
        elif self.join_method == 'sum':
            combined = embedding1 + embedding2

        output = self.relu(combined)
        output = self.dropout(output)
        output = self.fc(output)

        return output
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities of the target class
        pt = torch.exp(-CE_loss)

        # Compute the focal loss
        alpha_tensor = torch.tensor([self.alpha, 1-self.alpha]).to(inputs.device)
        alpha_t = alpha_tensor[targets.data.view(-1).long()].view(-1, 1)

        F_loss = alpha_t * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def model_factory(model_name, image_size, dropout=.2, num_classes=2, contrastive_mode = "None", contrastive_layer_size = 128, device="cuda", conv_layers = [32,64], fc_layers = [16], pretrained=True, path_to_weights="/local2/acc/MedicalNet_pretrained_branch/MedicalNet_pytorch_files2/pretrain/resnet_200.pth"):
    n_classes = contrastive_layer_size if contrastive_mode != "None" else num_classes
    if model_name == "3DCNN":
        model = Efficient3DCNN(in_channels=1, num_classes=n_classes, dropout_rate=dropout, conv_layers=conv_layers, fc_layers=fc_layers)
    elif model_name == "ViT":
        model = ViTWrapper(
            in_channels=1, 
            img_size=image_size, 
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
    elif model_name == "SEResNet152":
        model = monai.networks.nets.SEResNet152(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout)
    # elif model_name == "MedicalNet":
    #     model = MedicalNet(path_to_weights=path_to_weights, device=device,dropout_prob=dropout)
    #     for param_name, param in model.named_parameters():
    #         print(param_name)
    #         if param_name.startswith("fc"):
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = True
    elif model_name == "ResNet10":
        if pretrained:
            model = ResNetWrapper(
                model_name,
                dropout_prob=dropout,
                spatial_dims=3,
                n_input_channels=1,
                num_classes=n_classes, 
                pretrained=True,
                feed_forward=False, 
                shortcut_type="B",
                bias_downsample=False)
        else:
            raise ValueError("No training from scratch implemented")
    elif model_name == "ResNet18":
        if pretrained:
            model = ResNetWrapper(
                model_name,
                dropout_prob=dropout,
                spatial_dims=3,
                n_input_channels=1,
                num_classes=n_classes, 
                pretrained=True,
                feed_forward=False, 
                shortcut_type="A",
                bias_downsample=False
            )
        else:
            model = monai.networks.nets.resnet18(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=n_classes)

    else:
        raise ValueError(f"The model name {model_name} is not accepted.")
    

    if contrastive_mode != "None":
        return ContrastiveWrapper(model, contrastive_layer_size, num_classes, dropout_rate=dropout)
    else:
        return model 
    