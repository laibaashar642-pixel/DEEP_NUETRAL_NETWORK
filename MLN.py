""" #Multi Nuetral Network ye hamay tb use krna hota hai jb perception nai krna usmy sirf aik hi nueron hota hai jo sirf aik decision kai liye use hota hai so mln  isliya use hota hai kunkay jb humay multiple decisions krny hoty hai aur multiple nuerons ki zarurat hoti hai tu is mai forward propagation,backward propagation a jati hai 
#So, Forward Propagation means data ko agy bhejna 
#First is defining data First layer jiskay andr do hidden layers (2nuerons) hoty hai
import numpy as np
X=np.array([12,34,56,78])#X hamaray pass input hai 
actual_label=1#Aur ye jo actual label hai wo hamaray pass actual result ya value hai jo hamay pta hai kai asl value yhi hai
#Weights Define kiye mlp mai weights ek matrix hota hai nakay vector jb do hidden nuerons hai tu
weight1=np.array([[2.5,9.0,-8.3,9.6],
                 [-1.0,8,9.7,4.5]])
bias1=0.8
#Phir hum dusri hidden layer bnayain gai jis may inputs aur bias dalain gai
#W1 ek matrix hai kyunki ab 2 neurons hain jo 2 inputs se connected hain. Har row ek neuron ke weights hai
weight2=np.array([2,9])
bias2=0.3
#Then Hidden layer jahan sai actual kaam shur hota hai
z_hidden=np.dot(weight1,X)+bias1
H=np.maximum(0,z_hidden) #Relu
#Output Layer
z_out=np.dot(weight2,H)+bias2
output=1/(1+np.exp(-z_out))#Sigmoid function
print(f"Hidden_Layer Output(H):{H}")
print(f"z_out:{z_out:4f}")
print(f"Final Output:{output:4f}")
print(f"Decision:{'Pass'if output >= 0.5 else 'Fail'}")
error=actual_label-output
print(f"Error:{error:.4f}") """
""" Rule yaad rakho
Hidden neurons = 2  →  W2 mein 2 elements
Hidden neurons = 5  →  W2 mein 5 elements
Hidden neurons = 10 →  W2 mein 10 elements
W2 ka size hamesha = hidden neurons ki count. """
import numpy as np
X=np.array([12,43,78,20])
actual_label=1
weights=np.array([1,2.0,3.4,5.0],
                 [-1.0,3.2,4.4,9.0])
bias1=1.09
weights2=np.array([2,3])
bias2=9.0
#Then define the hidden layer
z_hidden=np.dot(weights,X)+bias1
#Then apply the relu
