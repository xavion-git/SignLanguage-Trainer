# 🤟 Sign Language Classifier

A neural network built from scratch with NumPy that recognizes American Sign Language (ASL) hand signs from image data.

## Overview

This project trains a two-layer neural network to classify images of ASL hand signs into 24 letter categories (A–Y, excluding J and Z which require motion). The model is implemented purely in NumPy — no deep learning frameworks — making it a great resource for understanding how neural networks work under the hood.

## Demo 
<img width="495" height="499" alt="Screenshot 2026-06-24 004447" src="https://github.com/user-attachments/assets/d226a23c-276f-419e-ab29-c08aaa9c63e1" />
<img width="484" height="480" alt="Screenshot 2026-06-24 004528" src="https://github.com/user-attachments/assets/b093b83f-fa9d-402d-9b82-a620c30740d7" />
<img width="482" height="497" alt="Screenshot 2026-06-24 004504" src="https://github.com/user-attachments/assets/7ca96956-4f28-4f66-885f-fab9f3c54011" />


## Dataset

Uses the [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset, a drop-in replacement for the classic MNIST benchmark.

- **Format:** CSV files with pixel values (28×28 grayscale images flattened to 784 features)
- **Classes:** 24 ASL letters (`A B C D E F G H I K L M N O P Q R S T U V W X Y`)
- **Training set:** `sign_mnist_train.csv`
- **Test set:** `sign_mnist_test.csv`

Place both files in a `sign_language_data/` folder at the project root.

## Model Architecture

```
Input (784)  →  Hidden Layer (25, ReLU)  →  Output Layer (25, Softmax)
```

- **Input:** 784 pixel values (normalized to 0–1)
- **Hidden layer:** 25 neurons with ReLU activation
- **Output layer:** 25 neurons with Softmax activation (one per class)
- **Training:** Gradient descent with backpropagation

## Project Structure

```
sign-language-classifier/
├── sign_language_data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── sign_language_classifier.ipynb
├── requirements.txt
└── README.md
```

## Requirements

```txt
numpy
pandas
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook `sign_language_classifier.ipynb` top to bottom. The main training entry point is:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)
```

After training, evaluate on the dev set:

```python
predictions = get_predictions(forward_prop(W1, b1, W2, b2, X_dev)[3])
print("Dev accuracy:", get_accuracy(predictions, Y_dev))
```

## How It Works

1. **Data loading** — CSVs are read into NumPy arrays and transposed so each column is one example
2. **Normalization** — pixel values divided by 255 to scale to [0, 1]
3. **Forward pass** — linear transform → ReLU → linear transform → Softmax
4. **Backward pass** — gradients computed analytically via backpropagation
5. **Parameter update** — weights and biases nudged in the direction that reduces loss
6. **Evaluation** — accuracy printed every 10 iterations during training

## Results

| Split | Accuracy |
|-------|----------|
| Train | TBD      |
| Dev   | TBD      |

*(Fill in after training)*

## Acknowledgements

- Dataset by [tecperson on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Inspired by Samson Zhang's MNIST from scratch tutorial
