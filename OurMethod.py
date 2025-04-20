import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import functional as F
import pytorch_lightning as pl
import math
from sklearn.metrics import average_precision_score
from torch.autograd import Variable

    

class Two_dimensional_Fast_Fourier_Transform(nn.Module):
    def __init__(self):
        super(Two_dimensional_Fast_Fourier_Transform, self).__init__()

    
    # Method to reshape tensor to square
    def reshape_to_square(self, tensor):
        # Get batch size B, channels C and number of elements N of the input tensor
        B, C, N = tensor.shape
        # Calculate the minimum square side length that can accommodate N elements
        side_length = int(np.ceil(np.sqrt(N)))
        # Calculate the total number of elements in the square tensor
        padded_length = side_length ** 2
        # Create a zero-padded tensor with shape (B, C, padded_length) on the same device as the input tensor
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        # Copy the input tensor to the first N positions of the padded tensor
        padded_tensor[:, :, :N] = tensor
        # Reshape the padded tensor to a square tensor with shape (B, C, side_length, side_length)
        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        # Return the square tensor, side length, side length and original element count
        return square_tensor, side_length, side_length, N

    # Method to filter frequency bands of the input tensor
    def filter_frequency_bands(self, tensor, cutoff=0.2):
        
    
        
        device = tensor.device
        # Convert input tensor data type to float
        tensor = tensor.float()
        # Call reshape_to_square method to convert input tensor to square tensor and get related information
        tensor, H, W, N = self.reshape_to_square(tensor)

        # Get batch size B and channel count C of the square tensor
        B, C, _, _ = tensor.shape

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1)).real

        return fft_tensor

    def forward(self, x):
        fft_tensor = self.filter_frequency_bands(x, cutoff=0.3)
        return fft_tensor



   
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * 
                   (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]
                
class ResNetFFT(pl.LightningModule):
    def __init__(self, lr, n_epoch):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.lr = lr
        self.n_epoch = n_epoch
        self.fft = Two_dimensional_Fast_Fourier_Transform()
        
        self.input_block = nn.Sequential(
            resnet.conv1,  # [B, 64, 128, 128]
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool              # [B, 64, 64, 64]
        self.encoder1 = resnet.layer1           # [B, 256, 64, 64]
        self.encoder2 = resnet.layer2           # [B, 512, 32, 32]
        self.encoder3 = resnet.layer3           # [B, 1024, 16, 16]
        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.IN_layer = nn.Sequential(
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        ) 
        
        

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)  # Binary classification: Real/AI-generated
        )

        self.classifier_skip = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)  # Binary classification: Real/AI-generated
        )
    
        self.out = nn.Linear(128, 2)
        
    def forward(self, x):
        # Encoder path
        x1 = self.input_block(x)       # [B, 64, 128, 128]
        x2 = self.pool(x1)             # [B, 64, 64, 64]
        x3 = self.encoder1(x2)         # [B, 256, 64, 64]
        x4 = self.encoder2(x3)         # [B, 512, 32, 32]
        x5 = self.encoder3(x4)         # [B, 1024, 16, 16]
        x5 = self.reduce1(x5)          # [B, 512, 16, 16]
        x5 = self.IN_layer(x5)         # [B, 512, 16, 16]
        batch_size, channels, height, width = x5.shape
        
        resnet_OUT = self.classifier_skip(x5)
        
        feat = x5.view(batch_size, channels, height * width)
       
        fft_tensor = self.fft(feat)
        print(fft_tensor.shape)
        # Classification
        output = self.classifier(fft_tensor)
        combined_output = torch.cat([output, resnet_OUT], dim=1)  
        final_output = self.out(combined_output)  
        return output

    def mmd_loss(self, source, target):
        def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            # First ensure input is 2D tensor
            if source.dim() > 2:
                source = source.view(source.size(0), -1)
            if target.dim() > 2:
                target = target.view(target.size(0), -1)
                
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)
            
            # Calculate Gaussian kernel
            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0 - total1) ** 2).sum(2)
            
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
            return sum(kernel_val)

        # Ensure input tensor dimensions are correct
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
            
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler}

   
    def training_step(self, batch, batch_idx):
        # Training step for binary classification
        img, label = batch
        logits = self(img)

        with torch.no_grad():
            # Get features for MMD
            x1 = self.input_block(img)
            x2 = self.pool(x1)
            x3 = self.encoder1(x2)
            x4 = self.encoder2(x3)
            x5 = self.encoder3(x4)
            x5 = self.reduce1(x5)
            x5 = self.IN_layer(x5)
            batch_size, channels, height, width = x5.shape
            feat = x5.view(batch_size, channels, height * width)

        # Calculate MMD loss (select images with label==0 and label==1 as source and target respectively)
        if (label == 0).sum() > 1 and (label == 1).sum() > 1:
            real_feat = feat[label == 0]
            fake_feat = feat[label == 1]
            
            # Get smaller batch size
            min_batch_size = min(real_feat.size(0), fake_feat.size(0))
            
            # Randomly select the same number of samples
            if real_feat.size(0) > min_batch_size:
                idx = torch.randperm(real_feat.size(0))[:min_batch_size]
                real_feat = real_feat[idx]
            if fake_feat.size(0) > min_batch_size:
                idx = torch.randperm(fake_feat.size(0))[:min_batch_size]
                fake_feat = fake_feat[idx]
                
            mmd = self.mmd_loss(real_feat, fake_feat)
        else:
            mmd = torch.tensor(0.0, device=self.device)

        # Cross entropy loss for binary classification with MMD regularization
        loss = F.cross_entropy(logits, label) + 0.2 * mmd
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        
        # Calculate F1 score, precision and recall
        tp = ((preds == 1) & (label == 1)).sum().float()
        fp = ((preds == 1) & (label == 0)).sum().float()
        fn = ((preds == 0) & (label == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # Calculate Average Precision (AP)
        probs = F.softmax(logits, dim=1)[:, 1]  # Get probability for positive class (AI-generated)
        labels_np = label.cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        try:
            ap = average_precision_score(labels_np, probs_np)
        except:
            ap = 0.0  # Handle possible exception cases
        
        # Record metrics
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_f1', f1, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train_precision', precision, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train_recall', recall, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train_ap', ap, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train_mmd', mmd, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        return {'loss': loss, 'acc': acc, 'f1': f1, 'ap': ap}

    def validation_step(self, batch, batch_idx):
        # Validation step for binary classification
        img, label = batch
        logits = self(img)
        
        # Cross entropy loss for binary classification
        loss = F.cross_entropy(logits, label)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        
        # Calculate F1 score, precision and recall
        tp = ((preds == 1) & (label == 1)).sum().float()
        fp = ((preds == 1) & (label == 0)).sum().float()
        fn = ((preds == 0) & (label == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # Record validation metrics
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1', f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_precision', precision, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('val_recall', recall, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}

    def test_step(self, batch, batch_idx):
        # Test step similar to validation step but for test dataset
        img, label = batch
        logits = self(img)
        
        # Cross entropy loss for binary classification
        loss = F.cross_entropy(logits, label)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        
        # Calculate F1 score, precision and recall
        tp = ((preds == 1) & (label == 1)).sum().float()
        fp = ((preds == 1) & (label == 0)).sum().float()
        fn = ((preds == 0) & (label == 1)).sum().float()
        tn = ((preds == 0) & (label == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        
        # Record test metrics
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_f1', f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_precision', precision, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_recall', recall, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_specificity', specificity, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1}


if __name__ == '__main__':
    model = ResNetFFT(None,None).to('cuda')
    input = torch.rand(4,3,224,224).to('cuda')
    out = model(input)
    