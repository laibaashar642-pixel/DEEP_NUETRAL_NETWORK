#Cost function ki madad sai hum loss calculate krty hai kai kitna actual loss aya hai hamaray network mai isy loss function bhi kha jata hai coost function different types kai hoty hai alag alag problems kai liye  Cost Function ek number deta hai (Loss), aur network ki koshish hoti hai is number ko jitna ho sake kam karna, training ke dauran. two types kai hoty hai mean squared error cross entropy mean squared error jb ap number predict kr rhy ho eg:ghr ki price predict krna yya temperature predict krna cross entropy jb ap class/category predict kr rhy ho(yes/no) ya multiple options hun its works with probability eg:spam,non spam ,billi/kutta,ghoda 
#Yaad rkhnay ka tareeqa activation function kai similar hai Number predict kar rahe ho? → MSE Category predict kar rahe ho? → Cross-Entropy Aap ne abhi tak MSE use kiya hai (kyunke output 0/1 tha aur aap squared error use kar rahe thay) — yeh chalta hai chote examples ke liye, lekin asal mein jab classification problem ho (Yes/No type), tab Cross-Entropy zyada behtar hoti hai MSE se.
# First is MSE(mean Squared Error)
import numpy as np
actual=1
output=0.7 #Network ka guess
mse_loss=(actual-output)**2
print("MSE_Loss:",mse_loss)
#Simple jitna output actual sai door hai,utna bara loss
#Cross Entorpy 
cross_entorpy_loss=-(actual * np.log(output)+(1-actual)*np.log(1-output))
print("Cross_Entropy_Loss:",cross_entorpy_loss)
#Then make comparison both of these cost functions
print("/n----- Comaprison-----")
for output_test in [0.9,0.5,0.1]:
    mse=(actual-output)**2
    ce=(actual*np.log(output_test)+(1-actual)*np.log(1-output_test))
    print(f"Output={output_test}MSE={mse:4f}CrossEntropy={ce:4f}")