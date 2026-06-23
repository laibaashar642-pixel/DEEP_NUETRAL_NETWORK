#Perceptron means kai jb hum pehli baar koi prediction krty hai apny nuerons ki help sai tu baaz dfa wo glt hoti hai lekin next time jb hum us cheez ko dekhty hai tu shi predict krty hai so usi ko perception kehty hai kai pehly wali gltyion sai seekhna.teen cheezain jo perception ko imp bnati hai one is step activation yani output 1 ya 0 kai inbetween hona chaye weights update hoty hai agr error 0 aye tu sb kuchshi hai aur agr error +1 aye tu weights badhao aur agr -1 ho tu glt weights ghatao aur inputs multiply hojaty hai
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
#Step Activation lgana chaye kai result 0 aur 1 kai drmayain mai ana chaye
output=1 if result1 >=0 else 0

print(f"Result1:{result1}")
print(f"Output:{output}")

#Weight update loop yani kai abhi tk hamari perception aik baar ho rhi hai sirf lekin weight update loop ki madad sai hum baar baaar hundred times perception krty hain aur apny result ko best sai best bnaty hai firstly make prediction,then extract the error then update the weights ye tb tk repeat kry gai jb tk 0 nai hojata ya fixed no of rounds complete na hojayain aur hr eik round ko epoch kehty hai step 1 is first make prediction and find output

import numpy as np
input1=np.array([45,78,90])
input2=np.array([23,45,33])
weight1=np.array([4,9,2])
weight2=np.array([9,3,1])
result=np.dot(input1,weight1)  + np.dot(input2,weight2)
#Inputs lety hai
bias=9.0
#Actual_label kia hota hai ye wo actual jawab hota hai jo hamay pehly sai hi pta ho yani actual score kia aye gha calculation  honey kai baad
weights=np.array([23,9,7.0,8])
actual_label=0.1
#Adding Bias
result1=result+bias
#Step Activation lgana chaye kai result 0 aur 1 kai drmayain mai ana chaye
output=1 if result1 >=0 else 0
error=actual_label-output
#Updation of weights
learning_rate=0.1
improved_weight=weight1+(error*input1*learning_rate)+weight2+(error*input2*learning_rate)
improved_bias=bias+(error*learning_rate)