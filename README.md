# Hopfield Network for Digit Restoration

## Overview

This project implements a Hopfield Network, a type of recurrent artificial neural network, to restore distorted handwritten digits from the MNIST dataset. By leveraging the principles of associative memory, the Hopfield Network can recall original images even when they are partially obscured or noisy.

## Table of Contents

- [Introduction](#introduction)
- [Project Focus](#project-focus)
- [Features](#features)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Hopfield Network was introduced by John Hopfield in 1982 and is inspired by the behavior of biological neural networks. It serves as a model for associative memory, where the network can retrieve full patterns from partial or distorted inputs. This project focuses on applying the Hopfield Network to restore images of handwritten digits from the widely used MNIST dataset.

## Project Focus

In this project, you will learn how to implement a Hopfield Network from scratch to demonstrate its capabilities in image restoration. The goal is to train the network using clean digit images, then test its performance by providing distorted versions of those images. You will explore how effectively the network can recover the original digits from noisy inputs.

### Key Objectives:

- Implement a Hopfield Network that can learn from multiple digit patterns.
- Distort images from the MNIST dataset by adding noise.
- Evaluate the network's ability to restore these images to their original forms.
- Measure the similarity between the original and restored images to assess the network's performance.

## Features

- **Image Restoration**: Restore distorted or noisy images of handwritten digits.
- **Visual Comparison**: Display original, noisy, and restored images side by side.
- **Similarity Measurement**: Calculate a similarity score to quantify restoration accuracy.
- **Python Implementation**: Easy to run and modify with clear code structure.

## Getting Started

### Prerequisites

To run this project, ensure you have Python and pip installed. You'll need the following libraries:

- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt

python main.py 
