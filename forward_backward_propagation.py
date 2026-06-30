#Forward Propagation bilkul similar hai usky jo abhi humnay single nueron mai prha hai like humnay input li agy bheji aur outputagya hai isi trah backward propagation mai jo hai is trah hota hai kai like agr hamaray forward propagation ki wja sai data glt arha hai ya error arha hai tu hum us data ko dubara wapis bhejty hai tu phir weights updation krty hai jissy improvement hoti hai
#hum aik chota sa hi network lain gai jiskay andr do inputs ati hai do hidden layers aur aik output
import numpy as np
X=np.array([1.2,3.4])
actual_label=1 #Correct Answer
w1=np.array([[0.9,0.8],
            [0.0,1.2]])#Hidden layer weights
b1=np.array([0.1,-0.5])#Hidden bias 
w2=np.array([0.6,-0.3])#output weights
b2=0.05 #Output Bias
#Calculation of hidden layer
z_hidden=np.dot(w1,X)+b1
#Applying activaation function relu
H=np.maximum(0,z_hidden)
#calculation of output layer
z_out=np.dot(w2,H)+b2
output=1/(1+np.exp(-z_out))#Then applying the sigmoid function
print("Hidden Layer(H):",H)
print("Network Guess (output):",output) 
# Yeh bilkul woh "Input → Hidden → Output" wala forward flow hai
#Now calculate the loss kai hamra guess kia hua answer kitna glt tha 
loss=(actual_label-output)**2
print("Loss:",loss)
#Now we have to move ulta chlana pry gha error ko correct krny kai liye this is backward_propagation
#output sai hidden ki traf wapis ana 
d_output=-(actual_label-output)*output*(1-output)
#Hidden layer tk glti ponchana
d_hidden=w2*d_output
d_hidden[z_hidden <=0]=0 #Relu
print("d_hidden:",d_hidden)
print("d_output:",d_output)
#then we aplly the activation function
""" Yeh "ulta chal ke pata lagana" wala part hai — d_output batata hai output kitna galat tha, d_hidden batata hai hidden layer ne us galti mein kitna hissa daala. """
"""
import numpy as np
X=np.array([1.2,3.4])
actual_label=1
w1=np.array([[1.2,0.3],  #Hidden layer weights
            [-0.3,1.1]])
b1=0.05#hidden layer bias
w2=np.array([0.4,1.3])#output layer weights
b2=4.6#Output Bias
#caluclation of hidden layer
z_hidden=np.dot(w1,X)+b1
#Applying activation function relu
H=np.maximum(0,z_hidden)
#Then calculating the output layer
z_out=np.dot(w2,X)+b2
#Applying activation function the sigmoid
output=1/(1+np.exp(-z_out))
print("Hidden_layer(H):",z_hidden)
print("Guessing the right(output):",output)"""