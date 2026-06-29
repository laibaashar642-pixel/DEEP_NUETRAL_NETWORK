#Training Loop means  the procees of teaching the network to get better through the repetition it built from the parts you know forward propagation,backward ,perception every single process or task you done to improve the network is called epoch like a simple summary is practice makes a person perfect
import numpy as np
#These are the four practice questions -like the four toys in our market
X=np.array([[0,0],
           [0,1],
           [1,0],
           [1,1]])
#These are the correct answers  for each practice questions
Y=np.array([0,1,1,0])
print(X)
print(Y)
#then we use or applying random weights to make a righ guessing of any number
np.random.seed(42)#keep the random numbers the same every time we run the code
#Hidden layer 2 inputs -> 2 hidden nuerons
w1=np.random.randn(2,2)#Weights for the hidden layer
b1=np.zeros(2)#Hidden Bias for hidden layer,starts at 0
#Output layer 2 hidden nuerons ->1 output
w2=np.random.rand(2)#Weights for output layer
b2=0.0 #bias for output layyer,starts at 0
print("w1",w1)
print("w2",w2)
print("b1",b1)
print("b2",b2)
"""  Making one single guess (forward pass)
Remember the market game — looking at one toy and guessing its price? That's what this is. We'll take just the first practice question (X[0], which is [0,0]) and make the network guess the answer. """
#Take first X example to test it for forward passing
x=X[0]#This is [0,0]
actual=Y[0]#This is 0 the correct answer
z_hidden=np.dot(w1,x)+b1
H=np.maximum(0,z_hidden) #Relu Activation
#Output Layer
z_out=np.dot(w2,H)+b2
output=1/(1+np.exp(-z_out))#Sigmoid Activation
print("Input:",x)
print("Hidden Layer Output(H):",H)
print("Network Guess(output)",output)
print("Correct Answer(actual):",actual)
#The next step is to guess how much your guessing correct or not means how wrong was our guess
loss=(actual-output) ** 2
print("Loss:",loss)
#backward pass infding the right direction to correct it
d_output=-(actual-output)*output*(1-output)
print("d_output:",d_output)
#Next step is to find how wrong your guessing and which direction we need to fix it
d_hidden=w2*d_output
d_hidden[z_hidden<=0]=0
print("d_hidden:",d_hidden)

#Turning the direction of mistake into real changes into the weights
learning_rate=0.5
#Update the output layer weights and bias
w2=w2-learning_rate*d_output*H
b2=b2-learning_rate*d_output

#Update hidden layers and weights
w1=w1-learning_rate*d_output*H
b1=b1-learning_rate*d_output
print("Updated_Weight:",w1)
print("Updated_Bias:",b1)
