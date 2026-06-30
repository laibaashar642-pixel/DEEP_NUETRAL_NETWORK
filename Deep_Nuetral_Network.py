""" Aap ne abhi tak jo network banaye, unmein sirf 1 hidden layer thi (2 neurons wali). "Deep" Neural Network ka matlab hai: 2 ya zyada hidden layers ek doosre ke peeche lagana.
Simple soch
Yaad karo factory wali example: 1 worker simple kaam kar sakta hai, lekin agar kai workers ek line mein ho (har ek apna kaam kar ke agle ko de), to woh zyada complex products bana sakte hain.
"Deep" ka matlab hai zyada workers ki lines — matlab zyada hidden layers, jo network ko zyada pechida (complex) patterns seekhne ki taqat deti hain.
Structure mein farq
Pehle (jo aap kar chuke ho):
Input → Hidden Layer (1) → Output
Ab (Deep Network):
Input → Hidden Layer (1) → Hidden Layer (2) → Output
(Ya isse bhi zyada layers ho sakti hain — 3, 4, 5...)
Kya naya seekhna padega?
Khushi ki baat yeh hai: concept bilkul wahi hai jo aap pehle se jaante ho — forward propagation, activation function, backward propagation. Sirf farq itna hai ke ek aur layer add ho rahi hai, isliye:

Forward pass mein ek extra step add hoga (Hidden Layer 1 → Hidden Layer 2)
Backward pass mein gradient ek aur layer ke through wapas jayega (chain thori lambi ho jayegi)

Bilkul wohi chain rule wala concept — bas ab chain mein ek aur link add ho gaya hai. 
Structure: Input (2) → Hidden Layer 1 (3 neurons) → Hidden Layer 2 (2 neurons) → Output (1)"""
import numpy as np

X = np.array([1.2, 3.4])     # input
actual = 1                    # sahi jawab

# Hidden Layer 1: 2 inputs -> 3 neurons
w1 = np.array([[0.9, 0.8],
               [0.0, 1.2],
               [0.5, -0.3]])
b1 = np.array([0.1, -0.5, 0.2])

# Hidden Layer 2: 3 inputs (Layer 1 se) -> 2 neurons
w2 = np.array([[0.6, -0.3, 0.4],
               [0.2, 0.9, -0.1]])
b2 = np.array([0.05, -0.2])

# Output Layer: 2 inputs (Layer 2 se) -> 1 output
w3 = np.array([0.7, -0.5])
b3 = 0.1

learning_rate = 0.5
# Hidden Layer 1
z1 = np.dot(w1, X) + b1
H1 = np.maximum(0, z1)        # ReLU

# Hidden Layer 2
z2 = np.dot(w2, H1) + b2
H2 = np.maximum(0, z2)        # ReLU

# Output Layer
z_out = np.dot(w3, H2) + b3
output = 1 / (1 + np.exp(-z_out))   # Sigmoid

print("H1:", H1)
print("H2:", H2)
print("Output:", output)
loss = (actual - output) ** 2
print("Loss:", loss)
# Output layer ka gradient
d_output = -(actual - output) * output * (1 - output)

# Hidden Layer 2 ka gradient (output se aaya hua asar)
d_H2 = w3 * d_output
d_H2[z2 <= 0] = 0              # ReLU rule

# Hidden Layer 1 ka gradient (Layer 2 se aaya hua asar)
d_H1 = np.dot(w2.T, d_H2)      # w2 transpose karna padta hai shape match karne ke liye
d_H1[z1 <= 0] = 0              # ReLU rule

print("d_output:", d_output)
print("d_H2:", d_H2)
print("d_H1:", d_H1)
w3 = w3 - learning_rate * d_output * H2
b3 = b3 - learning_rate * d_output

w2 = w2 - learning_rate * np.outer(d_H2, H1)
b2 = b2 - learning_rate * d_H2

w1 = w1 - learning_rate * np.outer(d_H1, X)
b1 = b1 - learning_rate * d_H1

print("\nUpdated weights done")