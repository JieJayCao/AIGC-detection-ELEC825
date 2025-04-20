import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, average_precision_score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from OurMethod import ResNetFFT
from dataset_loader import create_dataloaders

def test_model(model, test_loader, device, output_dir):
    
    model.eval()  # Set to evaluation mode
    print("Starting evaluation on test set...")
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():  # No gradient calculation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get prediction classes and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Collect prediction results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Assuming class 1 is positive (AI-generated)
    
    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test set accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")
    
    # Calculate Real Accuracy and Fake Accuracy
    # Real Accuracy: TP / (TP + FN) for Real Images (label 0)
    real_mask = (all_labels == 0)
    if np.sum(real_mask) > 0:
        real_accuracy = np.sum((all_preds == 0) & real_mask) / np.sum(real_mask)
        print(f"Real Accuracy: {real_accuracy:.4f}")
    else:
        real_accuracy = 0
        print("No Real Images samples")
    
    # Fake Accuracy: TP / (TP + FN) for Fake Images (label 1)
    fake_mask = (all_labels == 1)
    if np.sum(fake_mask) > 0:
        fake_accuracy = np.sum((all_preds == 1) & fake_mask) / np.sum(fake_mask)
        print(f"Fake Accuracy: {fake_accuracy:.4f}")
    else:
        fake_accuracy = 0
        print("No AI-generated image samples")
    
    # Calculate Average Precision (AP)
    ap = average_precision_score(all_labels, all_probs)
    print(f"Average Precision (AP): {ap:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Natural Image", "AI-generated Image"]))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Save ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    print(f"ROC curve saved to: {roc_path}")
    
    # Plot PR curve (Precision-Recall curve)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR Curve)')
    plt.legend(loc="lower left")
    
    # Save PR curve
    pr_path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    print(f"PR curve saved to: {pr_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["Natural Image", "AI-generated Image"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Display values on confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save results as text file
    results_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test set accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 score: {f1:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n")
        f.write(f"Real Accuracy: {real_accuracy:.4f}\n")
        f.write(f"Fake Accuracy: {fake_accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=["Natural Image", "AI-generated Image"]))
    
    print(f"Test results saved to: {results_path}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Test FFT model for detecting AI-generated images')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to save model')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--output_dir', type=str, default='./train_results', help='Path to save results')
    parser.add_argument('--log_path', type=str, default='fft_training_logs', help='Path to save logs')
    parser.add_argument('--base_save_name', type=str, default='fft_model', help='Base name for saved models')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--val_dir', type=str, help='Path to validation directory, if not specified will split from training set')
    parser.add_argument('--test_dir', type=str, help='Path to test directory, if not specified testing will not be performed')
    parser.add_argument('--test_only', action='store_true', help='Only perform testing, skip training')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for testing, only used when test_only is True')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNetFFT(lr=args.lr, n_epoch=args.epochs)
    model = model.to(device)
    
    # If only testing
    if args.test_only:
        if not args.checkpoint_path:
            print("Error: When setting --test_only, must provide --checkpoint_path")
            return
        
        if not args.test_dir:
            print("Error: When setting --test_only, must provide --test_dir")
            return
            
        # Load model
        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        model = ResNetFFT.load_from_checkpoint(args.checkpoint_path, lr=args.lr, n_epoch=args.epochs)        
        model = model.to(device)
        
        # Load test data
        test_loader = create_dataloaders(args.test_dir, batch_size=args.batch_size)
        
        # Perform testing
        test_model(model, test_loader, device, args.output_dir)
        return
    
    # Load dataset

    print(f"Loading training and validation sets from specified directories...")
    train_loader = create_dataloaders(args.data_dir, shuffle=True, batch_size=args.batch_size)
    val_loader = create_dataloaders(args.val_dir, shuffle=False, batch_size=args.batch_size)
    
    # Get dataset size information
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    dataset_size = train_size + val_size
    
    
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
   
    # Set up TensorBoard logger
    logger = TensorBoardLogger(save_dir=args.output_dir, name=args.log_path)
    print(f"TensorBoard logs will be saved to: {logger.log_dir}")
    
    # Set up early stopping and model checkpoint callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # Monitor validation loss
        dirpath=os.path.join(logger.log_dir, 'checkpoints'), # Save to checkpoints under log directory
        filename=f"{args.base_save_name}-{{epoch:02d}}-{{val_loss:.4f}}", # Filename format
        save_top_k=1, # Save the best model
        mode='min', # Lower metric is better
        save_last=True # Always save the last epoch model
    )
    
    # Initialize trainer
    print("Initializing Pytorch Lightning Trainer...")
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=[args.gpu_id] if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=1.0,
        # Log model structure and other information
        log_every_n_steps=50,
    )
    print("Trainer initialized.")

    # Start training
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")

    # Save final model
    final_model_path = os.path.join(logger.log_dir, f"{args.base_save_name}_final.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # If test directory is specified, load test set and perform testing
    if args.test_dir:
        print(f"Found test directory: {args.test_dir}")
        test_loader = create_dataloaders(args.test_dir, batch_size=args.batch_size)
        # Test using best model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model: {best_model_path}")
            best_model = ResNetFFT.load_from_checkpoint(best_model_path)
            best_model = best_model.to(device)
            test_accuracy = test_model(best_model, test_loader, device, args.output_dir)
            print(f"Accuracy of best model on test set: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
    # Train
    #  python train_fft.py --data_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/train --val_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/val
    
    # Test
    # python train_fft.py --data_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/StableDiffusion5/train --val_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/StableDiffusion5/val --test_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/StableDiffusion5/test --test_only --checkpoint_path /global/home/hpc5542/825/train_results/fft_training_logs/version_23/checkpoints/last.ckpt
    
    #python train_fft.py --data_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/train --val_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/val --test_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Fresh/BigGAN/test --test_only --checkpoint_path /global/home/hpc5542/825/train_results/fft_training_logs/version_22/checkpoints/fft_model-epoch=07-val_loss=3.4882.ckpt
    
    
    # benchmark 
    # python train_fft.py --data_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Benchmark/train --val_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Benchmark/val --test_dir /global/home/hpc5542/825/AIGCDetectBenchmark/dataset/Benchmark/test --test_only  --checkpoint_path ./train_results/fft_training_logs/version_25/fft_model_final.ckpt