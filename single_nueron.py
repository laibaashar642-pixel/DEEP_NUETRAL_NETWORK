#Inputs lety hai
import numpy as np
X=np.array([12,45,67,89])
#Weights lety hai
weights=np.array([1.2,4.8,9.0,8.7])
#Dot Product between inputs and weights
score=np.dot(X,weights)
bias=9.0
#Adding Bias
result=score+bias
#Activation Function
output=np.maximum(0,result)
print(f"Score:{score}")
print(f"Output:{output}")

