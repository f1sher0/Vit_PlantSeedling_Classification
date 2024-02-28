# Vit_PlantSeedling_Classification

## Project Description
This project focuses on the classification of plant seedling images using the Vision Transformer (ViT) architecture, showcasing its potential in image recognition tasks. By leveraging ViT, this project aims to accurately classify different species of plant seedlings, contributing to agricultural research and automation efforts.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation and Results](#model-evaluation-and-results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

## Installation
To set up this project, follow these steps:
1. Clone the repository to your local machine.
2. Ensure Python 3.x is installed.
3. Install the required dependencies: `pip install -r requirements.txt` (You may need to create this file listing all the necessary libraries, such as `torch`, `torchvision`, `matplotlib`, `PIL`, etc.)

## Usage
To use this project, run the following scripts in order:
1. `python makedata.py` to prepare the dataset.
2. `python train.py` to train the Vision Transformer model.
3. `python heatmap_plant.py` to generate attention heatmaps for model interpretation.

## Code Structure
- `makedata.py`: Prepares the dataset by organizing images into training and validation sets.
- `mean_std.py`: Calculates the mean and standard deviation of the dataset for normalization.
- `train.py`: Main script for training the Vision Transformer model.
- `heatmap_plant.py`: Generates attention heatmaps to visualize areas the model focuses on.
- `models/`: Contains the Vision Transformer model definitions.
- `data/`: Directory where the processed dataset is stored.

## Dataset
The dataset comprises various species of plant seedlings images. The `makedata.py` script splits the dataset into training and validation sets, ensuring a diverse representation for model training and evaluation.

## Model Training
The training process uses the Vision Transformer architecture, optimized with Adam and a Cosine Annealing LR scheduler. Model training is detailed in `train.py`, with data augmentation techniques like Cutout and Mixup to enhance model generalization.

## Model Evaluation and Results
Model evaluation is conducted using accuracy as the primary metric. The `heatmap_plant.py` script provides insight into the model's decision-making by visualizing attention heatmaps, overlaying areas of interest on the original seedling images.

## Contributing
Contributions to this project are welcome. Please submit an issue or pull request with your proposed changes or enhancements.

## License
This project is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Special thanks to the creators of the Vision Transformer architecture for their groundbreaking work in applying transformers to image recognition tasks.
- Gratitude to the agricultural research community for providing the plant seedlings dataset.

## Contact Information
For questions or support, please contact [Your Name] at [Your Email].
