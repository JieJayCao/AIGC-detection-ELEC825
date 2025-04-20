import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

LABEL_TO_INT = {
    "nature": 0, "ai": 1
}




class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        
        self.transform = transform
        self.data = []
        
        ai_path = os.path.join(img_dir, "ai")
        nature_path = os.path.join(img_dir, "nature")
        
        # Load AI generated images
        if os.path.exists(ai_path):
            ai_images = (glob.glob(os.path.join(ai_path, "*.png")) + 
                         glob.glob(os.path.join(ai_path, "*.PNG")) + 
                         glob.glob(os.path.join(ai_path, "*.jpg")) +
                         glob.glob(os.path.join(ai_path, "*.JPEG"))+
                         glob.glob(os.path.join(ai_path, "*.jpeg")))
            for img_path in sorted(ai_images):
                try:
                    # Try to open image to verify it's readable
                    Image.open(img_path)
                    self.data.append((img_path, LABEL_TO_INT["ai"]))
                except Exception as e:
                    print(f"Warning: Cannot read image {img_path}, skipped. Error: {str(e)}")
                    continue
        
        # Load natural images  
        if os.path.exists(nature_path):
            nature_images =(glob.glob(os.path.join(nature_path, "*.png")) + glob.glob(os.path.join(nature_path, "*.PNG")) + 
                         glob.glob(os.path.join(nature_path, "*.jpg")) +
                         glob.glob(os.path.join(nature_path, "*.JPEG"))+
                         glob.glob(os.path.join(nature_path, "*.jpeg")))
            for img_path in sorted(nature_images):
                try:
                    # Try to open image to verify it's readable
                    Image.open(img_path)
                    self.data.append((img_path, LABEL_TO_INT["nature"]))
                except Exception as e:
                    print(f"Warning: Cannot read image {img_path}, skipped. Error: {str(e)}")
                    continue
        
        # Add debug information
        print(f"Total number of images loaded: {len(self.data)}")
        print(f"Number of AI generated images: {sum(1 for _, label in self.data if label == LABEL_TO_INT['ai'])}")
        print(f"Number of natural images: {sum(1 for _, label in self.data if label == LABEL_TO_INT['nature'])}")
        
        if len(self.data) == 0:
            print(f"Warning: No image files found in {img_dir}")
            print(f"Directory contents: {os.listdir(img_dir)}")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image and label
        img_path, label = self.data[idx]
        clean_img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            clean_img = self.transform(clean_img)
            # Modify label conversion to avoid using single value for LongTensor initialization
            label = torch.tensor(label, dtype=torch.long)
        return clean_img, label


def create_dataloaders(img_dir, shuffle = True, batch_size=4):
    
    # Define transformations
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure fixed output size
    transforms.ToTensor()])     
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])])
    # Create dataset
    dataset = ImageDataset(img_dir, transform=transform)
    
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    return data_loader

# Usage example
def dataset_example():
    # Directory paths
    train_dir = "/global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/train"
    val_dir = "/global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/val"
    test_dir = "/global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/test"

    # Method 1: Create data loaders from separate directories
    train_loader = create_dataloaders(train_dir, batch_size=4)
    val_loader =  create_dataloaders(val_dir, batch_size=4)
    test_loader = create_dataloaders(test_dir, batch_size=4)
    
    for img, label in train_loader:
        print(img.shape, label.shape)
   
if __name__ == "__main__":
    dataset_example()