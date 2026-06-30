#Actiavation function help krta hai kai jo numbers hamaray pass aty hai in the form of raw unmai sai final answer ko select kr sky  ye bilkul aik filter or strainer ki trah act krta hai The easiest is relu its rule is agr number -ve hai tu usko 0 bna do aur agr +ve hai tu usko same wesy hi rehnay do agr ye actiavtion function nai hogha tu network bilkul aik seedhi line ki trah kaam kry gha aur real world kai data kai sth manage nai kr pye gha  activation function ek chhalni hai jo raw number ko decide karta hai ke aage kya bhejna hai, aur isi se network complex cheezain seekh
#Write three functions
import numpy as np
#We take data in the form of +ve,-ve or zero
x=np.array([-5,2,3,0,-9])
#Relu Activation function ki madad sai hum negative number ko 0 kr dain aur baki sb ko wesy hi rehnay dain gai
relu_output=np.maximum(0,x)
#Sigmoid means kai hr number ko 0 aur 1 kai drmyian dba do
sigmoid_output=1/(1+np.exp(-x))
#Tanh hr number ko -1 aur 1 kai drmyian dba do
tanh_output=np.tanh(x)
print("Input:",x)
print("Relu:",relu_output)
print("Sigmoid:",sigmoid_output)
print("Tanh:",tanh_output)