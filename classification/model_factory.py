import monai 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import SENet
from monai.networks.blocks.squeeze_and_excitation import SEResNeXtBottleneck
from classification.nilay_model import dual_paths
from classification.dual_vit import DualViT

class OCT3DCNNEncoder(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout_rate=.2):  # Adjust 'num_classes' based on your specific task
        super(OCT3DCNNEncoder, self).__init__()
        # Convolutional layers with specified kernel sizes and strides
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1)
        self.batch_norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=1, stride=2)
        
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm3d(32)
        
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.batch_norm3 = nn.BatchNorm3d(32)
        
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.batch_norm4 = nn.BatchNorm3d(32)
        
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        # self.batch_norm5 = nn.BatchNorm3d(32)

        # self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.batch_norm6 = nn.BatchNorm3d(32)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layer and softmax for classification
        self.fc = nn.Linear(32, num_classes)  # num_classes should be set to the number of output classes
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.pool1(self.conv1(x))))
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        
        x = F.relu(self.batch_norm4(self.conv4(x)))
        
        x = F.relu(self.conv5(x))
        # x = F.relu(self.batch_norm5(self.conv5(x)))

        # x = F.relu(self.batch_norm6(self.conv6(x)))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x  


    
class Efficient3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, conv_layers=[32,64], fc_layers=[], dropout_rate=0.5):
        super(Efficient3DCNN, self).__init__()

        # Conv3D layers
        layers = [nn.Conv3d(in_channels if i == 0 else conv_layers[i-1], 
                            conv_layers[i], kernel_size=3, stride=1, padding=1) 
                  for i in range(len(conv_layers))]
        self.conv_layers = nn.Sequential(*layers, nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)  # Correct placement

        # FC layers
        in_features = conv_layers[-1]
        self.fc_layers = nn.Sequential()
        for out_features in fc_layers:
            self.fc_layers.add_module('fc', nn.Linear(in_features, out_features))
            self.fc_layers.add_module('relu', nn.ReLU())
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
    def __init__(self, name, dropout_prob=.2, freeze=False, fc_layers=[], num_classes=2, *args, **kwargs):
        super(ResNetWrapper, self).__init__()
        if name == "ResNet10":
            self.resnet = monai.networks.nets.resnet10(*args, **kwargs)
            self.fc_size = 512
        elif name == "ResNet18":
            self.resnet = monai.networks.nets.resnet18(*args, **kwargs)
            self.fc_size = 512
        elif name == "ResNet50":
            self.resnet = monai.networks.nets.resnet50(*args, **kwargs)
            self.fc_size = 2048
        elif name == "ResNet101":
            self.resnet = monai.networks.nets.resnet101(*args, **kwargs)
            self.fc_size = 2048
        elif name == "ResNet200":
            self.resnet = monai.networks.nets.resnet200(*args, **kwargs)
            self.fc_size = 2048
        else:
            raise NotImplementedError("model not implemented")

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        in_features = self.fc_size
        self.fc_layers = nn.Sequential()
        for out_features in fc_layers:
            self.fc_layers.add_module('fc', nn.Linear(in_features, out_features))
            self.fc_layers.add_module('relu', nn.ReLU())
            self.fc_layers.add_module('dropout', nn.Dropout(dropout_prob))
            in_features = out_features
        self.fc_layers.add_module('fc_final', nn.Linear(in_features, num_classes))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        output = self.resnet(x)
        output = self.dropout(output)
        output = self.fc_layers(output)
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

def model_factory(
    model_name, 
    image_size,
    dropout=.2, 
    num_classes=2, 
    contrastive_mode = "None", 
    contrastive_layer_size = 128,
    conv_layers = [32,64], 
    fc_layers = [16], 
    pretrained=True, 
    freeze=False, 
    use_dual_paths=False, 
    patch_s=18, 
    hidden_s=768,
    mlp_d=3072,
    num_l=12,
    num_h=12,
    qkv=False,
):
    n_classes = contrastive_layer_size if contrastive_mode != "None" and model_name != "DualViT" else num_classes
    if model_name == "3DCNN":
        # model = Efficient3DCNN(in_channels=1, num_classes=n_classes, dropout_rate=dropout, conv_layers=conv_layers, fc_layers=fc_layers)
        model = OCT3DCNNEncoder(num_classes=n_classes, in_channels=1, dropout_rate=dropout)
    elif model_name == "ViT":
        model = ViTWrapper(
            in_channels=1, 
            img_size=image_size, 
            patch_size = (patch_s,patch_s,patch_s),
            hidden_size = hidden_s,
            mlp_dim = mlp_d,
            num_layers = num_l,
            num_heads = num_h,
            proj_type='conv', 
            classification=True, 
            post_activation = "None",
            dropout_rate = dropout,
            num_classes = n_classes,
            qkv_bias = qkv)
    elif model_name == "ResNext7":
        block = SEResNeXtBottleneck 
        layers = [1, 0, 0, 0] 
        groups = 8 # Number of groups for ResNeXt, typical values are 32 or 64
        reduction = 16 # Reduction ratio for SE module
        model = SENet(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout,
            block=block,
            layers=layers,
            groups=groups,
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False,
            reduction=reduction)
    elif model_name == "ResNext8":
        block = SEResNeXtBottleneck 
        layers = [1, 1, 0, 0] 
        groups = 16 # Number of groups for ResNeXt, typical values are 32 or 64
        reduction = 16 # Reduction ratio for SE module
        model = SENet(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout,
            block=block,
            layers=layers,
            groups=groups,
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False,
            reduction=reduction)
    elif model_name == "ResNext9":
        block = SEResNeXtBottleneck 
        layers = [1, 1, 1, 0] 
        groups = 16 # Number of groups for ResNeXt, typical values are 32 or 64
        reduction = 16 # Reduction ratio for SE module
        model = SENet(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout,
            block=block,
            layers=layers,
            groups=groups,
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False,
            reduction=reduction)        
    elif model_name == "ResNext10":
        block = SEResNeXtBottleneck 
        layers = [1, 1, 1, 1] 
        groups = 32 # Number of groups for ResNeXt, typical values are 32 or 64
        reduction = 16 # Reduction ratio for SE module
        model = SENet(
            spatial_dims=3,
            in_channels=1,
            num_classes=n_classes,
            dropout_prob=dropout,
            block=block,
            layers=layers,
            groups=groups,
            inplanes=64,
            downsample_kernel_size=1,
            input_3x3=False,
            reduction=reduction)
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
    elif model_name in ["ResNet10", "ResNet50", "ResNet101", "ResNet200"]:
        if pretrained:
            model = ResNetWrapper(
                model_name,
                dropout_prob=dropout,
                freeze=freeze,
                fc_layers=fc_layers,
                num_classes=n_classes, 
                spatial_dims=3,
                n_input_channels=1,
                pretrained=True,
                feed_forward=False, 
                shortcut_type="B",
                bias_downsample=False)
        else:
            if model_name == "ResNet10":
                model = monai.networks.nets.resnet10(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=n_classes)
            elif model_name == "ResNet50":
                model = monai.networks.nets.resnet50(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=n_classes)
            else:
                raise NotImplementedError("TODO implement")
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
    elif model_name == "DualViT":
        assert contrastive_mode != "None"
        model = DualViT(
            in_channels_1 = 1, 
            img_size_1 = image_size,
            patch_size_1 = (patch_s,patch_s,patch_s),
            in_channels_2 = 1,
            img_size_2 = image_size,
            patch_size_2 = (patch_s,patch_s,patch_s),
            hidden_size = hidden_s,
            mlp_dim = mlp_d,
            num_layers = num_l,
            num_heads = num_h,
            proj_type='conv', 
            classification=True, 
            post_activation = "None",
            dropout_rate = dropout,
            num_classes = n_classes,
            qkv_bias = qkv
        )
    else:
        raise ValueError(f"The model name {model_name} is not accepted.")
    

    if contrastive_mode != "None" and model_name != "DualViT":
        if use_dual_paths:
            return dual_paths(model, num_classes, dropout)
        return ContrastiveWrapper(model, contrastive_layer_size, num_classes, dropout_rate=dropout)
    else:
        return model 
    