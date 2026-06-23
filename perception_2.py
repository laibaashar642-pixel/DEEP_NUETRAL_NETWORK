import numpy as np
input1=np.array([45,78,90])
input2=np.array([23,45,33])
weight1=np.array([4,9,2])
weight2=np.array([9,3,1])
result=np.dot(input1,weight1)  + np.dot(input2,weight2)
#Inputs lety hai
bias=9.0

#Adding Bias
result1=result+bias
#Actual_label kia hota hai ye wo actual jawab hota hai jo hamay pehly sai hi pta ho yani actual score kia aye gha calculation  honey kai baad
actual_label=1
#Step Activation lgana chaye kai result 0 aur 1 kai drmayain mai ana chaye

output=1 if result1 >=0 else 0
error=actual_label-output
#Updation of weights
learning_rate=0.1
improved_weight1=weight1+(error*input1*learning_rate)
improved_weight2=weight2+(error*input2*learning_rate)
improved_bias=bias+(error*learning_rate)
print(f"Result1:{result1}")
print(f"Ouput:{output}")
print(f"Actual_Label:{actual_label}")
print(f"Error:{error}")
print(f"Improved_weight1:{improved_weight1}")
print(f"Improved_weight2:{improved_weight2}")
print(f"Improved_Bias:{improved_bias}")