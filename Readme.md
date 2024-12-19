Deepfake Detection Using GAN with ResNet32

This project implements a deepfake detection model based on a Generative Adversarial Network (GAN) architecture, using ResNet32 as the feature extractor in the discriminator. The model is trained and evaluated on a Kaggle image dataset.

Table of Contents

Project Overview

Features

Dataset

Model Architecture

Installation

Usage

Results

Future Work

Project Overview

Deepfakes pose a significant threat to the integrity of digital media. This project aims to develop a robust detection model capable of identifying manipulated images. The proposed solution leverages the power of GANs to effectively classify authentic and fake images.

Features

ResNet32 Integration: Utilized as the feature extractor in the GAN discriminator for high-quality feature representation.

Image-based Detection: Focused on static image detection.

GAN-based Training: Ensures the discriminator and generator evolve together for improved performance.

Scalable Implementation: Designed to handle diverse datasets for training and testing.

Dataset

![alt text](https://github.com/Sabal-Subedi/Deepfake/tree/main/images/dataset.png?raw=true)

Source: Kaggle

Description: A comprehensive dataset containing labeled real and fake images.

Preprocessing: Images were resized to 64x64 pixels and normalized for training.

Model Architecture

Generator

The generator creates synthetic images designed to mimic real images and deceive the discriminator.

Discriminator

The discriminator classifies images as real or fake. ResNet32 is employed to extract spatial features from the input images, enhancing the discriminator's performance.

Loss Functions

Generator Loss: Binary cross-entropy loss.

Discriminator Loss: Binary cross-entropy loss to distinguish real from fake images.

Installation

Prerequisites

Python 3.8+

GPU with CUDA support (optional but recommended)

Dependencies

Install the required dependencies using pip:

pip install -r requirements.txt

Clone the Repository

git clone https://github.com/Sabal-Subedi/Deepfake

Usage

Prepare the Dataset

Download the dataset from Kaggle and place it in the data/ directory.

Update the dataset path in the configuration file.

Train the Model

Evaluate the Model

Generate Visualizations


Results

![alt text](https://github.com/Sabal-Subedi/Deepfake/tree/main/images/confuse.png?raw=true)
Achieved high accuracy in detecting fake images.

![alt text](https://github.com/Sabal-Subedi/Deepfake/tree/main/images/loss.png?raw=true)
Loss curves and performance metrics are logged for analysis.

![alt text](https://github.com/Sabal-Subedi/Deepfake/tree/main/images/output.png?raw=true)

![alt text](https://github.com/Sabal-Subedi/Deepfake/tree/main/images/reacall.png?raw=true)


Future Work

Extend the model to handle video-based deepfake detection.

Incorporate temporal anomaly detection mechanisms.

Optimize for real-time detection on low-resource devices.
