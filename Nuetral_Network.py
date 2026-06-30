#Nuetral Network means aik insaan i dimag ki trah jb insaani dimag aik cheez ko baar baar dekhta hai tu usky pattern bnana shur kr deta hai isi trah nuetral network bhi mai bhi jb aik cheez dekhi jati hai tu wo patterns bnana shur kr deti hai is ki teen layers hoti hai input layer,hidden layer,output layer aur hr do layers kai drmyian jo connections hoty hai unko weights kehty hai 
#Inputs lety hai Yeh ek single neuron ka forward pass hai — bilkul wahi "Input → Soch → Output" wala flow, sirf yahan poori network ki jagah sirf ek neuron dikhaya gaya hai. Nuetral Network or single nueron aik hi cheez hai
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

