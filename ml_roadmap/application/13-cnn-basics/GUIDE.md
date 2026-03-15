# Lab 13: CNN Basics
> Convolution from scratch, then build and train CNNs on CIFAR-10, finishing with transfer learning

## Table of Contents
1. [Setup](#setup) - Dataset and libraries
2. [Visualize CIFAR-10](#step-1-visualize-cifar-10) - Understand the input
3. [Convolution from Scratch](#step-2-convolution-from-scratch) - 2D convolution with numpy
4. [Kernel Effects](#step-3-kernel-effects) - Edge detection, blur, sharpen
5. [Max Pooling from Scratch](#step-4-max-pooling-from-scratch) - Downsample
6. [PyTorch CNN](#step-5-pytorch-cnn) - Build the architecture
7. [Count Parameters](#step-6-count-parameters) - Why CNNs are efficient
8. [Train CNN](#step-7-train-cnn-on-cifar-10) - Full training loop
9. [MLP vs CNN](#step-8-mlp-vs-cnn) - Head-to-head comparison
10. [Transfer Learning](#step-9-transfer-learning) - Pretrained ResNet18
11. [Compare All](#step-10-compare-scratch-cnn-vs-pretrained) - Final accuracy table
12. [Visualize Filters](#step-11-visualize-filters) - What does the network see?

## Prerequisites
- Read `theory/13-cnn-basics/GUIDE.md` first
- Completed Lab 12 (PyTorch basics, training loops, nn.Module)
- Understanding of: convolution operation, feature maps, spatial hierarchy

## Dataset
**CIFAR-10** — 32x32 color images, 10 classes
- Via torchvision: `torchvision.datasets.CIFAR10`
- 50,000 train / 10,000 test
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Harder than Fashion-MNIST: color, more complex shapes, real photographs

## Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
```

---

## Step 1: Visualize CIFAR-10

1. Load the dataset:
   ```python
   transform = transforms.Compose([transforms.ToTensor()])
   train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
   ```
2. Show a grid of sample images (2 per class):
   ```python
   classes = train_data.classes
   fig, axes = plt.subplots(2, 5, figsize=(12, 5))
   for i, ax in enumerate(axes.flat):
       idx = next(j for j, (_, label) in enumerate(train_data) if label == i)
       img = train_data[idx][0].permute(1, 2, 0)  # CHW -> HWC
       ax.imshow(img)
       ax.set_title(classes[i])
       ax.axis('off')
   ```
3. Note: 32x32 is tiny. Images are low resolution. This is intentionally challenging.
4. Check shape: `(3, 32, 32)` — 3 color channels, 32x32 pixels

**Expected output:** Small but recognizable images. Colors matter for classification (green frog, blue sky for airplane).

---

## Step 2: Convolution from Scratch

Implement 2D convolution on a single grayscale image using numpy:

1. Convert one CIFAR image to grayscale:
   ```python
   img = train_data[0][0].numpy()         # (3, 32, 32)
   gray = np.mean(img, axis=0)            # (32, 32)
   ```

2. Define a 3x3 kernel (e.g., edge detection):
   ```python
   kernel = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]])
   ```

3. Implement convolution:
   ```python
   def convolve2d(image, kernel):
       h, w = image.shape
       kh, kw = kernel.shape
       pad_h, pad_w = kh // 2, kw // 2
       output = np.zeros_like(image)
       padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
       for i in range(h):
           for j in range(w):
               region = padded[i:i+kh, j:j+kw]
               output[i, j] = np.sum(region * kernel)
       return output
   ```

4. Apply and visualize: original vs convolution output side by side

**Expected output:** Edge detection kernel highlights edges in the image. The output shows bright lines where intensity changes sharply.

> **Checkpoint 1:** Your scratch convolution should produce visible edge-detected output.

---

## Step 3: Kernel Effects

Apply different kernels to the same image and show results:

1. **Edge detection** (already done above)
2. **Blur (box filter):**
   ```python
   blur_kernel = np.ones((3, 3)) / 9
   ```
3. **Sharpen:**
   ```python
   sharpen_kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])
   ```
4. **Horizontal edge detection (Sobel):**
   ```python
   sobel_h = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
   ```

5. Show all 4 outputs in a 2x2 grid with titles

**Expected output:** Blur smooths the image. Sharpen makes details crisper. Sobel highlights horizontal edges specifically. CNNs learn these kernels automatically.

---

## Step 4: Max Pooling from Scratch

1. Implement 2x2 max pooling:
   ```python
   def max_pool2d(image, pool_size=2):
       h, w = image.shape
       out_h, out_w = h // pool_size, w // pool_size
       output = np.zeros((out_h, out_w))
       for i in range(out_h):
           for j in range(out_w):
               region = image[i*pool_size:(i+1)*pool_size,
                              j*pool_size:(j+1)*pool_size]
               output[i, j] = np.max(region)
       return output
   ```

2. Apply to the edge-detected image
3. Show: original (32x32) -> convolution (32x32) -> pooling (16x16)
4. Note: pooling halves spatial dimensions while keeping the strongest features

**Expected output:** Pooled output is 16x16, retaining the most prominent edges. Spatial detail is reduced but key features remain.

---

## Step 5: PyTorch CNN

Build a simple CNN:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x16 -> 16x16
        self.pool = nn.MaxPool2d(2, 2)                             # halves spatial dim
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 32x32 -> 16x16, 16 channels
        x = self.pool(self.relu(self.conv2(x)))   # 16x16 -> 8x8, 32 channels
        x = x.view(x.size(0), -1)                 # flatten: 32*8*8 = 2048
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Trace the shapes through the network on paper. Verify with a test input:
```python
model = SimpleCNN()
test_input = torch.randn(1, 3, 32, 32)
print(model(test_input).shape)  # should be (1, 10)
```

**Expected output:** Output shape (1, 10) — one prediction per class.

> **Checkpoint 2:** You understand how spatial dimensions change through conv -> relu -> pool layers.

---

## Step 6: Count Parameters

1. Count parameters at each layer:
   ```python
   for name, param in model.named_parameters():
       print(f"{name}: {param.numel():,}")
   ```
2. Compare CNN parameter count with an equivalent MLP:
   - MLP: 3072 (input) * 128 (hidden) = 393,216 just in first layer
   - CNN conv1: 3 * 16 * 3 * 3 + 16 = 448 parameters

3. Key insight: convolution shares weights across spatial positions. This is why CNNs are so much more efficient than MLPs for images.

**Expected output:** CNN has far fewer parameters than an MLP of equivalent depth. Conv layers have hundreds of params; FC layers have thousands.

---

## Step 7: Train CNN on CIFAR-10

1. Prepare data with augmentation:
   ```python
   transform_train = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   ```

2. Train for 20 epochs:
   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()
   ```

3. Track both training and validation loss/accuracy each epoch
4. Plot: epoch vs loss (train + val) and epoch vs accuracy (train + val)

**Expected output:** Test accuracy around 70-75% after 20 epochs for the simple CNN. Training accuracy will be higher (some overfitting expected).

---

## Step 8: MLP vs CNN

1. Build an MLP with similar parameter count:
   ```python
   class FlatMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.flatten = nn.Flatten()
           self.fc1 = nn.Linear(3072, 512)
           self.fc2 = nn.Linear(512, 128)
           self.fc3 = nn.Linear(128, 10)
           self.relu = nn.ReLU()

       def forward(self, x):
           x = self.flatten(x)
           x = self.relu(self.fc1(x))
           x = self.relu(self.fc2(x))
           x = self.fc3(x)
           return x
   ```
2. Train MLP on same CIFAR-10 data for same epochs
3. Compare test accuracy: CNN vs MLP
4. Why CNN wins: spatial structure, weight sharing, translation invariance

**Expected output:** MLP gets ~50-55% accuracy. CNN gets ~70-75%. The gap is large and clear. MLP treats pixels as independent features; CNN understands spatial relationships.

> **Checkpoint 3:** CNN significantly outperforms MLP on image data. You understand why.

---

## Step 9: Transfer Learning

Use a pretrained ResNet18 and fine-tune for CIFAR-10:

1. Load pretrained model:
   ```python
   resnet = models.resnet18(pretrained=True)
   ```

2. Replace the final layer (ResNet18 outputs 1000 classes for ImageNet):
   ```python
   resnet.fc = nn.Linear(resnet.fc.in_features, 10)
   ```

3. Optionally freeze early layers:
   ```python
   for param in resnet.parameters():
       param.requires_grad = False
   resnet.fc.requires_grad_(True)
   ```

4. Train for 10 epochs with a small learning rate (0.001)
5. Then unfreeze all layers, train 5 more epochs with lr=0.0001

**Expected output:** Fine-tuned ResNet18 reaches ~90-93% accuracy — much better than the scratch CNN. Transfer learning leverages features learned on ImageNet.

---

## Step 10: Compare Scratch CNN vs Pretrained

Create a summary table:

```
| Model          | Test Accuracy | Parameters | Training Time | Epochs |
|----------------|--------------|------------|---------------|--------|
| MLP            | ~52%         |            |               | 20     |
| Simple CNN     | ~73%         |            |               | 20     |
| ResNet18 (ft)  | ~92%         |            |               | 15     |
```

Key takeaways:
- Architecture matters more than training time
- Pretrained features transfer well between datasets
- Simple CNNs work but deeper networks are much better

**Expected output:** Clear progression from MLP to CNN to pretrained. Each jump is significant.

---

## Step 11: Visualize Filters

1. Extract first conv layer weights:
   ```python
   filters = model.conv1.weight.data.cpu()
   ```
2. Plot the learned 3x3 filters as small images:
   ```python
   fig, axes = plt.subplots(4, 4, figsize=(8, 8))
   for i, ax in enumerate(axes.flat):
       if i < filters.shape[0]:
           f = filters[i].permute(1, 2, 0)  # to HWC
           f = (f - f.min()) / (f.max() - f.min())  # normalize to [0,1]
           ax.imshow(f)
       ax.axis('off')
   ```
3. Compare with the hand-crafted kernels from Step 3 — do any learned filters resemble edge detectors?

**Expected output:** First layer filters often learn edge detectors, color detectors, and gradient patterns — similar to the hand-crafted kernels but learned automatically from data.

> **Checkpoint 4:** You've built CNNs from scratch convolution to transfer learning. You understand the full pipeline and why each architectural choice matters.

---

## Summary Deliverables
- [ ] 2D convolution from scratch with visible kernel effects
- [ ] Max pooling from scratch
- [ ] PyTorch CNN trained on CIFAR-10
- [ ] MLP vs CNN comparison (CNN wins by ~20%)
- [ ] Transfer learning with pretrained ResNet18
- [ ] Accuracy comparison table across all approaches
- [ ] Visualized learned filters
