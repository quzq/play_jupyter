#%%
import numpy as np
x = np.arange(10)
print(x)

#%%
import matplotlib.pyplot as plt
year = [1980, 1985, 1990, 2000, 2010, 2018]
weight = [3, 15, 25, 55, 62, 58]
plt.plot(year, weight)
plt.show()
#%%
import pandas as pd
pd.DataFrame([[1, "abc"], [2, "def"]], columns=("id", "name"))


#%%
