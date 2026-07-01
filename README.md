# Deep Neural Network — From Scratch in NumPy

This repo is my hands-on journey of learning how neural networks actually work under the hood — **no TensorFlow, no PyTorch, just NumPy**. Har concept pehle samajh kar likha hai, phir code kiya hai, taake "black box" na lagay balke pura pata ho ke andar ho kya raha hai.

The files are organized roughly in the order I learned them: starting from a single neuron, moving to perceptrons that learn from mistakes, then full networks with forward + backward propagation, cost functions, activation functions (ReLU, Sigmoid, Softmax), the calculus behind backprop (chain rule), a deep network with multiple hidden layers, and finally a full training loop over many epochs.

---

## 📁 File-by-File Breakdown

### 🔹 `single_nueron.py`
The very first building block — a **single neuron**. Takes a set of inputs, multiplies them by weights, adds a bias, and passes the result through a ReLU activation function. This is the "atom" that every neural network is built from.

### 🔹 `Nuetral_Network.py`
Basically the same single-neuron forward pass as above, but written to explain the bigger picture: a neural network is just this same idea (input → weighted sum → activation → output) repeated and stacked across **input, hidden, and output layers** connected by weights.

### 🔹 `perception.py`
Introduces the **Perceptron** — a neuron that learns from its mistakes. Covers:
- Step activation (output 0 or 1)
- Calculating the **error** (actual − predicted)
- Updating weights and bias based on that error using a learning rate
- The full **weight-update loop** repeated over multiple rounds (epochs) until the error shrinks

### 🔹 `perception_2.py`
A cleaner, focused rewrite of the perceptron learning rule — takes inputs, makes a prediction, computes the error against the actual label, and updates each weight and the bias individually using the learning rate.

### 🔹 `Cost_Function.py`
Explains **how a network measures how wrong it is**. Implements and compares two loss functions:
- **Mean Squared Error (MSE)** — used for regression problems (predicting a number, e.g. price)
- **Cross-Entropy Loss** — used for classification problems (predicting a category, e.g. spam/not spam)

### 🔹 `Activation_Function.py`
Implements the three core activation functions from scratch:
- **ReLU** — turns negative numbers to 0, keeps positive numbers as-is (default choice for hidden layers)
- **Sigmoid** — squashes any number between 0 and 1 (used for binary yes/no output)
- **Tanh** — squashes any number between −1 and 1

Also includes a simple decision guide for **when to use which activation function** in the output layer depending on the type of prediction (binary, multi-class, or continuous number).

### 🔹 `MLN.py` (Multi-Layer Network)
Moves from a single neuron to a proper **Multi-Layer Perceptron** with one hidden layer containing multiple neurons. Introduces the idea that weights become a **matrix** (not just a vector) once you have more than one neuron per layer, and walks through a full forward pass: input → hidden layer (ReLU) → output layer (Sigmoid).

### 🔹 `forward_backward_propagation.py`
Puts **forward propagation** and **backward propagation** together in one small network (2 inputs → hidden layer → 1 output). Shows how the network:
1. Makes a guess (forward pass)
2. Calculates the loss
3. Sends the error backward through the network to figure out each layer's contribution to the mistake (backward pass)

### 🔹 `backward_propagation.py`
A focused, clean version of full backward propagation — forward pass, loss calculation, computing gradients (`d_output`, `d_hidden`) using the Sigmoid and ReLU derivatives, and then **updating all the weights and biases**.

### 🔹 `Derivative_of_backward_propagation.py`
The "why" behind backpropagation — derives the backprop formula from scratch using the **chain rule**. Breaks the calculation into three linked pieces (Loss → Output → z_out → weight) and shows how multiplying these three "links" together gives you the gradient, instead of just using the formula blindly.

### 🔹 `Backward_Propagation for softmax and sigmoid.py`
Extends backpropagation to **multi-class classification** using the **Softmax** activation function (for when there are more than 2 possible categories, e.g. cat/dog/bird). Explains how Softmax converts raw scores into probabilities that sum to 1, and shows that when Softmax is paired with Cross-Entropy loss, the backward pass simplifies beautifully to just `predicted probabilities − actual one-hot label`.

### 🔹 `Deep_Nuetral_Network.py`
The jump from a shallow network (1 hidden layer) to a **Deep Neural Network** with **two hidden layers** stacked back-to-back. Same core concepts (forward pass, ReLU, Sigmoid, backward pass, weight updates) — just one more layer added to both the forward and backward chains.

### 🔹 `Training_Nuetral_Network.py`
Sets up the classic **XOR problem** dataset (4 input pairs with their correct outputs) and walks through **one single training step**: forward pass on one example, calculate loss, backward pass, and update the weights once.

### 🔹 `Training_loop.py`
The final piece — takes everything above and wraps it into a real **training loop**: runs the full forward → loss → backward → weight-update cycle across **all 4 XOR examples, repeated for 1000 epochs**, so the network actually practices and improves over time instead of learning from just one example once.

---

## 🧠 What I Learned

- How a single neuron computes: `output = activation(weights · inputs + bias)`
- How a perceptron learns from its errors and updates its own weights
- The difference between MSE and Cross-Entropy loss, and when to use each
- ReLU, Sigmoid, Tanh, and Softmax activation functions — what they do and when to use them
- Full forward propagation across multiple layers
- Full backward propagation, including deriving the gradient formulas using the **chain rule**
- How to extend backprop to multi-class problems using Softmax + Cross-Entropy
- Building a **Deep Neural Network** with multiple hidden layers
- Writing a complete **training loop** that trains a network over many epochs

All of this is implemented **from scratch using only NumPy** — no deep learning frameworks — to build a solid foundation in how neural networks actually work before moving on to libraries like TensorFlow/PyTorch.

---

## 🛠️ Tech Used
- Python
- NumPy

## ▶️ How to Run
Each file is a standalone script. Just run any file directly:
```bash
python Training_loop.py
```