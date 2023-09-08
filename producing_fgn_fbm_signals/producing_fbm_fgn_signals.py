# -*- coding: utf-8 -*-
"""producing_FBM_FGN_signals.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qk0_Qu8PmzwvQLoA1ZPDnFOS8u4T9ymL
"""

pip install fbm

from fbm import FBM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd

#data set stands
time_series=FBM(n=1024, hurst=0.75, method='daviesharte')

# Generate a fBm realization
fbm_dataset = time_series.fbm()

# Generate a fGn realization
fgn_dataset = time_series.fgn()

# Get the times associated with the fBm
t_values = time_series.times()    #devided (0,1) in n parts

print(type(fbm_dataset),type(fgn_dataset),type(t_values))
print(fbm_dataset.shape,fgn_dataset.shape,t_values.shape)

plt.plot(t_values,fbm_dataset, label="FBM")
plt.plot(t_values[:-1],fgn_dataset, label="FGN")

plt.legend()

#data set stands
time_series=FBM(n=1024, hurst=0.9, method='daviesharte')

# Generate a fBm realization
fbm_dataset9 = time_series.fbm()

#data set stands
time_series=FBM(n=1024, hurst=0.5, method='daviesharte')

# Generate a fBm realization
fbm_dataset5 = time_series.fbm()
#data set stands
time_series=FBM(n=1024, hurst=0.2, method='daviesharte')

# Generate a fBm realization
fbm_dataset2 = time_series.fbm()
# Get the times associated with the fBm
t_values = time_series.times()    #devided (0,1) in n parts

plt.figure(figsize=(15,5))
plt.plot(t_values,fbm_dataset2, label="H=0.2")
plt.plot(t_values,fbm_dataset5, label="H=0.5")
plt.plot(t_values,fbm_dataset9, label="H=0.9")
plt.xlabel("t")
plt.ylabel("x(H,t)")
plt.legend()

"""#Test section"""

import numpy as np
np.arange(0.05,1,0.1)

#produce and save  FBM and FGN signals
hurst_exponent=np.arange(0.05,1,0.1)
all_fbm=[]
all_fgn=[]

for h in tqdm(hurst_exponent):
  all_fbm=[]
  all_fgn=[]
  for i in range(100):#number of ensemble
    #data set stands
    time_series=FBM(n=1024, hurst=h, method='daviesharte')

    # Generate a fBm realization
    fbm_dataset = time_series.fbm()
    all_fbm.append(fbm_dataset )

    # Generate a fGn realization
    fgn_dataset = time_series.fgn()
    all_fgn.append(fgn_dataset)
    # Get the times associated with the fBm
    t_values = time_series.times()    #devided (0,1) in n parts

  with open('/content/drive/MyDrive/cod/producing_fgn_fbm_signals/data/FGN'+str(round(h,2))+'.npy', 'wb') as f:
    np.save(f, np.array(all_fgn))
  with open('/content/drive/MyDrive/cod/producing_fgn_fbm_signals/data/FBM'+str(round(h,2))+'.npy', 'wb') as f:
    np.save(f, np.array(all_fbm))

#read data
data_FBM=np.load(open('/content/drive/MyDrive/cod/data_fgn_fbm/FBM0.05.npy', 'rb'))
data_FGN=np.load(open('/content/drive/MyDrive/cod/data_fgn_fbm/FGN0.05.npy', 'rb'))