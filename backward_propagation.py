import numpy as np
#Data Fillup
X=np.array([1.2,3.4])
w1=np.array([[0.9,0.8],
            [0.0,1.2]])
b1=np.array([0.1,-8.6])
w2=np.array([0.6,-4.5])
b2=0.05
learning_rate=0.5
#Forward Pass
z_hidden=np.dot(w1,X)+b1
H=np.maximum(0,z_hidden)
z_out=np.dot(w2,H)+b2
output=1/(1+np.exp(-z_out))
print(f"Output:{output:.4f}")
#backward Propagation kia  hota hai  yani kai jb apni glti ya errors ko improve krna ho weights ko badha kr ya km kar kai loss calculate krky ya phir gradient calculate krky
#So for backward propagation the first step is calculation of loss
actual_label=1
#Loss
loss=(actual_label-output) ** 2
print(f"Loss:{loss:4f}")
#Now the main step is backward propagation where we add output gradient (sigmoid derivative)
d_ouput=-(actual_label-output)*output*(1-output)
print(f"d_output:{d_ouput:.6f}")
#Hidden Gradient Relu Derivative
d_hidden=w2*d_ouput
d_hidden[z_hidden<=0]=0
print(f"d_hidden:{d_hidden}")
#Weights Updation
w2=w2-learning_rate*d_ouput*H
b2=b2-learning_rate*d_ouput
w1=w1-learning_rate*np.outer(d_hidden,X)
b1=b1-learning_rate*d_hidden
print(f"\n=== Updated Weights ===")
print(f"New Weight W2 {w2}")
print(f"New Weight W1 {w1}")
print(f"New base2 {b2:.6f}")
print(f"New base 1{b1}")
