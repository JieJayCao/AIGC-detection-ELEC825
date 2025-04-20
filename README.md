# AIGC-detection-ELEC825

## Spectral-Spatial Fusion Network for AI-Generated Image Detection

It is the 📘 Final Project for ELEC 825 (Machine Learning/Deep Learning W25), Queen’s University, Kingston, ON, Canada. This project implements an AI-generated image detection model based on ResNet and FFT (Fast Fourier Transform). The model effectively distinguishes between natural images and AI-generated images.

## Requirements

- Python 3.8+
- CUDA support (recommended for GPU training)
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `OurMethod.py`: Model implementation containing ResNetFFT model definition
- `dataset_loader.py`: Data loader implementation
- `train_ourmethod.py`: Training and testing script
- `requirements.txt`: Project dependencies

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── ai/
│   └── nature/
├── val/
│   ├── ai/
│   └── nature/
└── test/
    ├── ai/
    └── nature/
```

## Usage

### Dataset
The dataset used in this project is from [GenImage](https://github.com/GenImage-Dataset/GenImage), a million-scale benchmark for detecting AI-generated images. You can get a simple sample dataset from this link [Dataset](https://github.com/GenImage-Dataset/GenImage).


### Training
```bash
python train_ourmethod.py \
    --data_dir /path/to/train/dataset \
    --val_dir /path/to/val/dataset \
    --batch_size 200 \
    --epochs 10 \
    --lr 0.0005
```

### Testing

```bash
python train_ourmethod.py \
    --data_dir /path/to/train/dataset \
    --val_dir /path/to/val/dataset \
    --test_dir /path/to/test/dataset \
    --test_only \
    --checkpoint_path /path/to/checkpoint.ckpt
```

## Command Line Arguments

- `--data_dir`: Path to training dataset
- `--val_dir`: Path to validation dataset
- `--test_dir`: Path to test dataset
- `--batch_size`: Batch size (default: 200)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0005)
- `--test_only`: Run testing only
- `--checkpoint_path`: Path to model checkpoint
- `--output_dir`: Output directory (default: ./train_results)
- `--gpu_id`: GPU ID (default: 0)

## Model Features

The model incorporates the following key features:
- ResNet50-based feature extraction
- FFT feature analysis
- MMD (Maximum Mean Discrepancy) loss
- Instance Normalization

## Model Output

The model generates the following evaluation metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC Curve
- Confusion Matrix
- AP (Average Precision)

All results will be saved in the specified output directory.

## Training Logs

Training logs and model checkpoints will be saved in:
```
./train_results/fft_training_logs/
```
## Comparison and SOTA methods
This repository compares various methods for AI-Generated Content (AIGC) detection, including state-of-the-art (SOTA) techniques. We utilize the [AIGCDetectBenchmark repository](https://github.com/Ekko-zn/AIGCDetectBenchmark) to evaluate and test these methods alongside our own approach.

## Important Notes

1. Ensure sufficient GPU memory is available
2. Large batch sizes are recommended for training
3. Training performance can be optimized by adjusting learning rate and batch size
4. The best model will be automatically saved during training
5. Early stopping is implemented to prevent overfitting


## 👨‍💻 Contributors

This project was collaboratively completed by the following Queen’s University graduate students:
Team Member Contributions

| Member              | Estimated Contribution | Contributions                                                                                     |
|---------------------|-------------------------|----------------------------------------------------------------------------------------------------|
| **Jie Cao**         | 25%                     | **Report**: Methodology  <br> **Development**: Proposed method development, training, and testing |
| **Nicholas Chivaran** | 25%                   | **Report**: Results & Analysis, Ethical Implications  <br> **Development**: Dataset preparation, reference method integration and testing |
| **Henry Yuan**      | 25%                     | **Report**: Experimental Setup, Conclusion  <br> **Development**: Dataset preparation, reference method integration and testing |
| **Zelin Zhang**     | 25%                     | **Report**: Abstract, Introduction, Related Works  <br> **Development**: Proposed method development, training, and testing |
