""" #Inputs lety hai
import numpy as np
inputs=np.array([12,45,67,89])
#Weights lety hai
weights=np.array([1.2,4.8,9.0,8.7])
#Dot Product between inputs and weights
score=np.dot(inputs,weights)
bias=9.0
#Adding Bias
result=score+bias
#Activation Function
output=np.maximum(0,result)
print(f"Score:{score}")
print(f"Output:{output}")
 """
import numpy as np
input=np.array([12,34,16,19])
#Applying the weights
weights=np.array([1.2,4.0,8.9,2])
#Applying dot product
score=np.dot(input,weights)
bias=0.9
result=bias+score
#Activation function

output=np.maximum(0,result)
#Then print the output and score
print(f"Score:{score}")
print(f"Output:{output}")