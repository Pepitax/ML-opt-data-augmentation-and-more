# ML-opt-data-augmentation-and-more

# **Description**
We explore the impact of different optimizers and varying sub-training dataset sizes to identify the most effective optimizer for tasks involving limited data.  
Subsequently, we apply data augmentation techniques to these small datasets to evaluate improvements in classification accuracy.  
The applied for our exploration is classification of the cifar10 dataset. 


# 📁 **Organization of Files**

The main file containing all plots and results is: **`run.ipynb`**

---

### 🧪 **Notebook Structure**
- **Part 1**: Optimizer performance exploration using various sub-training dataset sizes, along with PCA-based 2D loss landscape visualizations for insight.
- **Part 2**:
  - Calls `train_model_cifar100.py` to train a CNN backbone on CIFAR-100 for transfer learning.
  - Executes multiple `.py` scripts to train on CIFAR-10 using different data augmentation techniques.
  - Runs transfer learning experiments using the CIFAR-100 backbone.

---

### 📄 **File Descriptions**

- **`run.ipynb`**: The main notebook for running experiments and visualizing results.
- **`train_model_cifar100.py`**: Trains the backbone CNN model on CIFAR-100, used for transfer learning.

### 🧱 **Data Augmentation Scripts**
Each script trains a CNN with a specific data augmentation technique:
- `data_augmentation_flip.py`: Applies horizontal flipping.
- `data_augmentation_crop.py`: Applies random cropping.
- `data_augmentation_color_jittered.py`: Applies color jittering (brightness, contrast, saturation, hue).
- `data_augmentation_rotation.py`: Applies random rotation.
- `data_augmentation_flip&crop.py`: Combines flipping and cropping.
- `data_augmentation_flip&crop&friends.py`: Combines multiple augmentations: flip, crop, jitter, and rotation.

### 🔁 **Transfer Learning Scripts**
Use a pretrained CIFAR-100 backbone model:
- `data_augmentation_flipwCifar100.py`: Applies flipping and initializes convolutional weights from the CIFAR-100 model (classifier weights are randomly reinitialized).
- `project_wCifar100.py`: Baseline script using the CIFAR-100 pretrained backbone, no augmentation.

### 🧩 **Base Script**
- `project_base.py`: The base training script executed over 10 random seeds.  
  **Note**: All other `.py` files are derived from this file with added data augmentation or transfer learning capabilities.
