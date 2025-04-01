# Neural Network from Scratch (NumPy)

This project implements a simple **feedforward neural network** from scratch using Python and NumPy, without any high-level machine learning libraries like TensorFlow or PyTorch.

You’ll find everything from forward and backward propagation to parameter updates using stochastic gradient descent (SGD), plus experiments on model performance with different hyperparameters.

---

## 📁 Project Structure

```plaintext
.
├── neural_network.py        # Core neural network implementation
├── additional_code.py       # Runs experiments with plots
├── autograder.py            # Grades model using test cases
├── test_utils.py            # Helper functions for testing and validation
├── data/
│   ├── smallTrain.csv
│   ├── smallValidation.csv
│   ├── mediumTrain.csv
│   ├── mediumValidation.csv
│   ├── largeTrain.csv
│   └── largeValidation.csv
└── test_cases/              # Contains test case files (if applicable)
```

---

## Features

- Manual implementation of:
  - Linear layers with bias
  - Sigmoid activation
  - Softmax classification
  - Cross-entropy loss
  - Forward and backward propagation
  - SGD-based weight updates
- Handles multi-class classification
- Visualizes model performance over time

---

## 📊 Datasets

- Located in `/data/`
- Files:  
  - `smallTrain.csv`, `smallValidation.csv`  
  - `mediumTrain.csv`, `mediumValidation.csv`  
  - `largeTrain.csv`, `largeValidation.csv`

These contain binary features and digit labels (0–9).

---

## 🔬 Experiments

The `additional_code.py` script performs two key experiments:

1. **Hidden Units Experiment**  
   Varies the number of hidden units and plots the effect on training and validation loss.

2. **Learning Rate Experiment**  
   Compares different learning rates to observe convergence behavior.

---

## ▶️ How to Run

### 1. Set up Environment

Make sure you have Python 3 installed. Then install dependencies:

```bash
pip install numpy matplotlib
```

### 2. Prepare Data

Place your CSV datasets in a `data/` folder structured like this:

```
/data
  ├── smallTrain.csv
  ├── smallValidation.csv
  ├── mediumTrain.csv
  ├── mediumValidation.csv
  ├── largeTrain.csv
  └── largeValidation.csv
```

> You can edit the file paths in `neural_network.py` if your data is stored elsewhere.

### 3. Run Experiments

To run the learning rate and hidden units experiments with plots:

```bash
python additional_code.py
```

You’ll see cross-entropy loss visualizations plotted via `matplotlib`.

### 4. Run Autograder (Optional)

If you want to test the correctness of your implementation using test cases:

```bash
python autograder.py
```

---

## ⚙️ Configuration

Inside `additional_code.py`, you can modify:

```python
hidden_units_list = [5, 20, 50, 100, 200]
learning_rates = [0.1, 0.01, 0.001]
epochs = 50
```

Feel free to tweak these values to test model performance under different settings.

---

## 📌 Requirements

- Python 3.x
- NumPy
- Matplotlib

---

## 📌 Notes

- All layers and gradients are calculated manually using NumPy.
- This project is educational—ideal for understanding the fundamentals of neural networks.
- The model uses one hidden layer and trains using full SGD (no batching).

---

## 💡 Future Improvements

- Add support for ReLU activation
- Include dropout or regularization
- Experiment with different initializations
- Add multi-layer (deep) architecture
- Support batch or mini-batch training

---

## 👨‍💻 Author

This project was built to demonstrate and reinforce understanding of how neural networks work under the hood—no black-box libraries involved.

---

## 🧪 Sample Output

After running `additional_code.py`, you’ll see graphs like:

- Training vs Validation Loss across epochs
- Loss comparison across different learning rates
- Impact of hidden unit count on model performance
