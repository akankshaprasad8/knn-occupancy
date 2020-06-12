
import json
from decimal import Decimal
import pandas as pd

jsondata = '{"number": 1.573937639}'
array=r'{"AMG8833": {"0_16":["24.50",25.00,24.50,24.25,24.75,24.50,24.75,25.00,25.25,24.75,24.75,24.75,25.50,25.00,25.25,25.25],"16_32":[25.00,25.00,25.25,25.50,25.25,25.25,25.25,26.00,25.50,25.00,25.00,25.50,25.50,25.25,25.75,25.75],"32_48":[25.50,25.50,25.00,25.25,25.00,26.00,26.00,26.25,26.25,25.25,25.00,25.50,25.25,25.75,25.75,26.75],"48_64":[26.25,26.00,26.00,25.75,26.00,25.75,26.25,27.25,27.75,27.00,26.00,26.50,25.75,27.00,27.00,28.00]}}'
list=[]
for i in x["AMG8833"]["0_16"]:
  list.append(float(i))
for i in x["AMG8833"]["16_32"]:
  list.append(i)  
for i in x["AMG8833"]["32_48"]:
  list.append(i)  
for i in x["AMG8833"]["48_64"]:
  list.append(i)  

df = pd.DataFrame([[list]]) 