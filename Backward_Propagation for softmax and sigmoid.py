""" Aap ne abhi tak jo kiya, woh Yes/No type ka tha (XOR — 0 ya 1). Wahan Sigmoid sahi tha.
Ab naya scenario: 2 se zyada options
Socho aap ka network yeh decide kar raha hai: "yeh tasveer billi hai, kutta hai, ya parinda hai?" — yahan 3 options hain, sirf 2 nahi.
Sigmoid yahan kaam nahi karega kyunke woh sirf ek number 0-1 ke darmiyan deta hai — lekin humein 3 alag probabilities chahiye, jo mil kar 100% (1.0) banayein.
Softmax kya karta hai
Softmax ek aisa activation function hai jo kai numbers leta hai aur unhe probabilities mein convert kar deta hai — taake sab mil kar 1 (ya 100%) ban jayein.
Simple soch: socho 3 dostoon ne ek competition mein scores banaye: 5, 2, 1. Softmax in scores ko le kar kehta hai "pehle dost ke jeetne ke chances 70%, doosre ke 20%, teesre ke 10%" — total milake 100%.
Jis dost ka score sabse zyada tha, usay sabse zyada probability milti hai — lekin baqi dono ko bhi thori probability milti hai (zero nahi hoti, chahe kitna bhi chota score ho).
Formula (simple version)
Har number ko e (exponential) ki power mein le jate hain, phir sab ko jor kar har ek ko us total se divide karte hain:
softmax(xi)=exi∑exj\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}softmax(xi​)=∑exj​exi​​
Plain English: "har number ko bara karo (exponential se), phir dekho woh total ka kitna hissa hai."
Backward propagation mein farq
Jab Softmax use karte hain output layer mein, to uske sath usually Cross-Entropy loss use hoti hai (jo humne Topic 4 mein dekhi thi). Achi baat yeh hai ke jab Softmax + Cross-Entropy dono sath use karte hain, to unka combined backward propagation formula bohot simple ban jata hai:
d_output=predicted probabilities−actual (one-hot) labelsd\_output = \text{predicted probabilities} - \text{actual (one-hot) labels}d_output=predicted probabilities−actual (one-hot) labels
Matlab koi lambi calculation nahi — sirf predicted minus actual, seedha subtract. """
#For Softmax Activation Function
import numpy as np
#Think you have three categories one is cat, second is dog, third is sparrow
#Ye raw scores hai z_out jaisy activation function lganay sai pehly
scores=np.array([5,2,1])
#Softmax formula hr number ko exponential mai lejao then divide by total
exp_scores=np.exp(scores)
probabilities=exp_scores/np.sum(exp_scores)
print("Raw_scores:",scores)
print("Exp_scores:",exp_scores)
print("Probabilities:",probabilities)
print("Total should be 1.0:"np.sum(probabilities))
scores = np.array([5, 2, 1]) 
""" — yeh teen categories ke raw numbers hain (jitna bara number, utna confident network us category ke liye)
exp_scores = np.exp(scores) — har number ko e ki power mein le jate hain. Yeh sab numbers ko positive bana deta hai aur bare numbers ko aur bara kar deta hai (taake winner clearly nazar aaye)
probabilities = exp_scores / np.sum(exp_scores) — har exp_score ko total se divide kar dete hain, taake sab mil kar 1.0 (100%) banayein
Akhri print confirm karta hai ke teeno probabilities ka total 1.0 hai """
#Now backward propagation(Softmax+cross entropy)
#Let's think kai first category is cat (first_category,index 0)
actual_one_hot=np.array([1,0,0]) #One hot encoding[1=cat,0=dog,0=sparrow]
#Backward Propagation jaisa humnay discuss kia  yah seedha subtract krna 
d_output=probabilities-actual_one_hot
print("\n Predicted probabilities:",probabilities)
print("\nActual One Hot:",actual_one_hot)
print("\nd_output(gradient):",d_output)
""" actual_one_hot — sahi jawab ko is tarah likha jata hai: jo category sahi hai uske liye 1, baqi sab 0. Isay one-hot encoding kehte hain.
d_output = probabilities - actual_one_hot — bas seedha subtract, jaisa humne theory mein dekha. Jahan network ka guess sahi tha (high probability for billi), wahan gradient chota hoga. Jahan galat tha, wahan bara hoga. """