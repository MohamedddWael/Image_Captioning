# Image Captioning System with Transfer Learning and RNN
This repository contains an implementation of an Image Captioning System using Transfer Learning for feature extraction and a Recurrent Neural Network (RNN) for caption generation. The project is built using the Flickr8k dataset and aims to generate natural language descriptions for images.

## Project Overview
This project is divided into two phases:

Feature Extraction: Using a pre-trained CNN model (VGG16) to extract high-level image features.
Caption Generation: Using a simple LSTM-based RNN model to generate captions for images.
The system is trained and evaluated on the Flickr8k dataset, with a focus on achieving over 85% accuracy on the validation dataset and generating relevant captions in real-time.

## Dataset
Flickr8k Dataset:
The Flickr8k dataset contains 8,000 images and their corresponding captions. It is available on Kaggle: Flickr8k Dataset.

## Dataset Preprocessing:
Resized and preprocessed images to fit the input requirements of VGG16.
Cleaned captions by:
Removing special characters and punctuations.
Converting text to lowercase.
Adding <start> and <end> tokens.
Explored the dataset:
Counted word frequencies and caption lengths.
Built a vocabulary with frequently occurring words.
## Implementation Details
Model Architecture
Encoders:

Image Encoder:
Used VGG16 as the feature extractor.
Passed the extracted image features through a fully connected layer with ReLU activation and batch normalization to reduce dimensionality.
Text Encoder:
Used an LSTM layer to encode sequential features from tokenized captions.
Aggregation Layer:

Combined the output of the two encoders into a single vector carrying both image and caption information.
Decoder:

Used a fully connected layer to map the aggregated vector to the vocabulary size.
Applied a softmax layer for word prediction.
Loss Function:

Used categorical cross-entropy loss to optimize the model.
Evaluation Metric:

Monitored accuracy during training.
Used evaluation metrics like BLEU score and CIDEr for generated captions.
## Training Details
Data Augmentation: Applied random augmentations to enhance the training data.
Dataset Split:
Training Set: 7591 images
Validation Set: 250 images
Test Set: 250 images
Optimizer: Adam
Hyperparameter Tuning: Tuned learning rate and dropout to improve generalization.
Regularization: Added dropout layers to prevent overfitting.
