""" Sabse pehle: hum yeh kar kyun rahe hain?
Aap ne pehle backprop ka formula use kiya tha (d_output = -(actual - output) * output * (1 - output)), lekin sirf use kiya — yeh samjhe bina ke yeh formula banta kahan se hai.
Aaj hum yeh dekhenge ke yeh formula khud-ba-khud kahan se nikal aata hai.
Chain wali soch — dobara, simple example se
Socho 3 cheezain hain jo ek line mein judi hain:
A → B → C
Matlab: A, B ko affect karta hai, aur B, C ko affect karta hai. A seedha C ko touch nahi karta — uska asar B se hoke guzarta hai.
Sawal: Agar A thora badle, to C kitna badlega?
Jawab: Do cheezain nikal kar multiply karo:

B kitna badha jab A badla
C kitna badha jab B badla

In dono ko multiply kar do — yehi jawab hai. Isay Chain Rule kehte hain.
Ab apne network mein yehi chain dhoondhte hain
Aap ke network mein bhi ek aisi hi chain hai:
w2 (weight) → z_out → output → Loss

w2 seedha Loss ko touch nahi karta
w2 pehle z_out banata hai
z_out se output banta hai
output se Loss nikalta hai

Toh agar humein janna hai "w2 ko kitna badalna chahiye taake Loss kam ho," humein teen "asar" nikal kar multiply karne padenge:

Loss, output ke badalne se kitna badalta hai
output, z_out ke badalne se kitna badalta hai
z_out, w2 ke badalne se kitna badalta hai

Teeno ko multiply karne se humein pata chal jata hai w2 ka Loss pe poora asar. """
#The summary is jb teeno cheezain aik linne mai judi hon yani chain ki form mai hun tu unka asr nikalanay kai liye hr link ka asr nikal kr multiply krty hai
import numpy as np
#Same to same we set data like we did first
X=np.array([1.2,3.4])
actual=1
w1=np.array([[0.9,0.8],
             [0.0,1.2]])
b1=np.array([0.1,-0.5])
w2=np.array([0.6,-0.3])
b2=0.05
#Forward pass (Guess Bnana)
z_hidden=np.dot(X,w1)+b1
H=np.maximum(0,z_hidden)
z_out=np.dot(w2,H)+b2
output=1/(1+np.exp(-z_out))
print("Output_Guess: ",output)
#Now we extract the three links one by one
#First Link loss output kai change sai kitna badalta hai 
link1=-2*(actual-output)
print("Link1(Loss vs Output):",link1)
#Second Link output z_out kai change sai kitna badalta hai
link2=output*(1-output)
print("Link2(Output vs z_out):",link2)
#Third Link z_out ,w2 kai change sai kitna badalta hai
link3=H
print("Link3 (z_out vs w2):",link3)

#Simple Summary 
""" link1 — agar output thora badle, Loss kitna badlega
link2 — Sigmoid ka apna khaas formula, output khud isay nikal sakta hai
link3 — z_out, seedha H ke barabar badalta hai (kyunke z_out = w2 × H + b2, ek linear relation hai) """
#Then multiply this three links
d_w2=link1*link2*link3
print("\n Final: Loss vs w2(d_w2):",d_w2)