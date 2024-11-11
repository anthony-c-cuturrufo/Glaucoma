import torch 
import torch.nn as nn
from classification.DenseNet3D import DenseNet3D
from torchvision import models

def recursive_weight_transfer(model2d, model3d):
    for name2d, module2d in model2d.named_children():
        # Check if the corresponding module exists in the 3D model
        if hasattr(model3d, name2d):
            module3d = getattr(model3d, name2d)
            # Compare functionality rather than direct types
            if isinstance(module2d, nn.Conv2d) and isinstance(module3d, nn.Conv3d):
                # print(f"Transferring weights for Conv layer: {name2d}")
                transfer_weights(module2d, module3d)
            elif isinstance(module2d, nn.BatchNorm2d) and isinstance(module3d, nn.BatchNorm3d):
                # print(f"Transferring weights for BatchNorm layer: {name2d}")
                transfer_weights(module2d, module3d)
            else:
                # If not a direct match, attempt to recursively transfer weights within nested modules
                recursive_weight_transfer(module2d, module3d)
        else:
            print(f"Module {name2d} not found in 3D model.")

def transfer_weights(layer2d, layer3d):
    with torch.no_grad():
        if isinstance(layer2d, nn.Conv2d) and isinstance(layer3d, nn.Conv3d):
            depth = layer3d.weight.shape[2]
            if layer3d.weight.shape[1] == 1:
                weight_2d_avg = layer2d.weight.mean(dim=1, keepdim=True)
                expanded_weight = weight_2d_avg.unsqueeze(2).repeat(1, 1, depth, 1, 1)
            else:
                expanded_weight = layer2d.weight.unsqueeze(2).repeat(1, 1, depth, 1, 1)
            layer3d.weight.copy_(expanded_weight)
            if layer2d.bias is not None:
                layer3d.bias.copy_(layer2d.bias)
        elif isinstance(layer2d, nn.BatchNorm2d) and isinstance(layer3d, nn.BatchNorm3d):
            layer3d.weight.copy_(layer2d.weight)
            layer3d.bias.copy_(layer2d.bias)
            layer3d.running_mean.copy_(layer2d.running_mean)
            layer3d.running_var.copy_(layer2d.running_var)



def densenet121_3D(pre_trained=None, num_classes=1000):
    """
    Create a 3D DenseNet-161 model and optionally transfer pre-trained weights from the 2D DenseNet-161.
    
    Args:
        pre_trained (bool): Whether to transfer pre-trained weights from the 2D DenseNet-161.
        num_classes (int): Number of output classes.

    Returns:
        model3d (DenseNet3D): 3D DenseNet-161 model.
    """
    model3d = DenseNet3D(growth_rate=32, block_config=[6, 12, 24, 16], num_init_features=64, num_classes=num_classes)
    
    if pre_trained:
        print('[IFNO]: Load pretrained weights')
        # Load pre-trained weights from the 2D DenseNet-161 model
        model2d = models.densenet121(weights="DEFAULT")
        
        # Transfer weights from 2D to 3D model
        recursive_weight_transfer(model2d.features, model3d.features)

    return model3d

