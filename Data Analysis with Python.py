#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 100%%; height: 50px; margin: 10px 0 0 0"/>
# 
# <h2 id="overview" style="width: 40%; height: 100px; margin: 10px 0 0 30%; color: rgba(255,255,255, 0.9);text-shadow: -1px 0 black, 0 1px black, 1px 0 black, 0 -1px black; background-image: url(https://images.unsplash.com/photo-1517524416126-7b0f7a174589?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=756&q=80);background-repeat: no-repeat;background-position: center;background-size: cover; padding: 40px 0 0 0; font-size: 30px; text-align:center; font-weight: bolder; border-radius:50%">Overview of Covered Content</h2>

# In[383]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sales = pd.read_csv('sales_data.csv', parse_dates = ['Date'])


# In[3]:


sales.head()


# In[392]:


get_ipython().system('head sales_data.csv')
#both get me the head of the data, but the first pandas (pd) organized it and presented it in a readable format
#this is used to checkout the data before importing it!


# In[16]:


#how many rows and columns we have?
sales.shape


# In[18]:


#to quickly understand the columns we are working with:
sales.info()


# In[19]:


sales.describe()


# In[20]:


sales['Unit_Cost'].describe()


# In[21]:


sales['Unit_Cost'].mean()


# In[22]:


sales['Unit_Cost'].median()


# ## Plotting with matplotlib, directly from pandas (will be explained in the pandas lessons)

# In[24]:


#Box Plot
sales['Unit_Cost'].plot(kind = 'box', vert = False, figsize=(14,6))


# In[25]:


#Density Plot
sales['Unit_Cost'].plot(kind = 'density', figsize=(14,6))


# In[26]:


ax = sales['Unit_Cost'].plot(kind = 'density', figsize = (14,6))
ax.axvline(sales['Unit_Cost'].mean(), color = 'red')
ax.axvline(sales['Unit_Cost'].median(), color = 'green')


# In[28]:


ax = sales['Unit_Cost'].plot(kind='hist', figsize = (14,6))
ax.set_ylabel('Number of Sales')
ax.set_xlabel('Dollars')


# In[29]:


sales.head()


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>

# ## Categorical analysis and visualization
# 
# > Analyzing Age_Group column

# In[30]:


sales['Age_Group'].value_counts()


# In[33]:


sales['Age_Group'].value_counts().plot(kind='pie', figsize = (6,6))


# In[34]:


ax = sales['Age_Group'].value_counts().plot(kind='bar', figsize=(14,6))
ax.set_ylabel('Number of Sales')


# ## Relationship between the columns

# In[5]:


corr = sales.corr()
corr


# In[6]:


fig = plt.figure(figsize=(8,8))
plt.matshow(corr, cmap='RdBu', fignum=fig.number)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
plt.yticks(range(len(corr.columns)), corr.columns);


# In[7]:


sales.plot(kind='scatter', x='Customer_Age', y='Revenue', figsize=(6,6))


# In[8]:


sales.plot(kind='scatter', x='Revenue', y='Profit', figsize = (6,6))


# In[9]:


ax = sales[['Profit', 'Age_Group']].boxplot(by='Age_Group', figsize=(10,6))
ax.set_ylabel('Profit')


# In[10]:


boxplot_cols = ['Year', 'Customer_Age', 'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit']
sales[boxplot_cols].plot(kind='box', subplots=True, layout=(2,3), figsize = (14,8))


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>

# ## Importing and Exporting Data
# 
# the main library in Python is MatPlotLib
# 
# We will:
# > get data from public API

# In[1]:


import requests
import pandas as pd


# In[2]:


def get_historic_price(symbol, exchange='bitfinex', after='2018-08-01'):
    url = 'https://api.cryptowat.ch/markets/{exchange}/{symbol}usd/ohlc'.format(symbol=symbol, exchange=exchange)
    resp = requests.get(url, params = {
        'periods':'3600',
        'after': str(int(pd.Timestamp(after).timestamp()))
    })
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data['result']['3600'], columns=['CloseTime', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'NA'])
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit='s')
    df.set_index('CloseTime', inplace=True)
    return df


# ### Pulling data from Bitcoin and Ether, for the last 7 days:

# In[3]:


last_week = (pd.Timestamp.now() - pd.offsets.Day(7))
last_week


# In[4]:


btc = get_historic_price('btc', 'bitstamp', after=last_week)


# In[5]:


eth = get_historic_price('eth', 'bitstamp', after=last_week)


# In[6]:


btc.head()


# In[7]:


eth.head()


# In[8]:


eth['ClosePrice'].plot(figsize=(15,7))


# In[9]:


btc['ClosePrice'].plot(figsize=(15,7))


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>

# ### Dynamic plots with Bokeh
# 
# > Interactive plots that can be manipulated right from the browser!

# In[10]:


from bokeh.plotting import figure, output_file, show

from bokeh.io import output_notebook


# In[11]:


output_notebook()


# In[12]:


p1 = figure(x_axis_type='datetime', title='Crypto Prices', width=800)
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'

p1.line(btc.index, btc['ClosePrice'], color='#f2a900', legend='Bitcoin')

p1.legend.location = 'top_left'

show(p1)


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>

# ### Exporting to Excel

# In[14]:


# creating an excel writer, a component of the pandas package

writer = pd.ExcelWriter('cryptos.xlsx')


# In[15]:


#writing both our bitcoin and ether data as separate sheets
btc.to_excel(writer, sheet_name='Bitcoin')


# In[16]:


eth.to_excel(writer, sheet_name='Ether')


# In[17]:


writer.save()


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 100%; height: 50px; margin: 10px 0 0 0"/>
# 
# <h2 id="numPyIntroduction" style="width: 25%; height: 300px; margin: 10px 0 0 30%; color: rgba(255,255,255, 0.9);text-shadow: -1px 0 black, 0 1px black, 1px 0 black, 0 -1px black; background-image: url(https://matplotlib.org/stable/_images/sphx_glr_voxels_numpy_logo_001.png);background-repeat: no-repeat;background-position: center;background-size: cover; padding: 40px 0 0 0; font-size: 30px; text-align:center; font-weight: bolder; border-radius:20%">Introduction to NumPy</h2>
# 

# ## Numpy Introduction
# 
# lets say you have a table with a variable Age (ranging from 0 to 120)
# 
# Another column is Dollars (range from 0-60billions)
# 
# 
# > Even though they are integers (plain number) => they have different annotations, thus different requirements in storage size
# 
# We can do the maths of how many bits we need to store age:
# 
# 2**7 = 128 => if we have 7 bits, we are going to store from 0 up to: 1 1 1 1 1 1 1 => this is equal to 127 in decimal + 0 = 128
# 
# So for the age, the memory we will be using is 7 bits per user
# 
# >> What if we have to use millions?
# 
# 2**32 => ... with 32 bits, we can store that (depending on the data and the size => just an example here)
# 
# > What if you have the entire population of the earth? => every bit is going to be important, because it will take a ton of data
# 
# => Numpy allows you to be very efficient in selecting the current number of bits ( 8 bits is 1 byte)
# 
# ### Numpy is a library that has a very advanced numeric processing that allows you to select the number of bits to use from memory to process data

# In[20]:


# in numpy, you can create number and control the size it has in bits:
np.int8


# In[21]:


np.int16


# ## Numpy is an Array processing library
# 
# > 99% about constantly processing arrays
# 
# 
# #### You cannot reply on advanced CPU directive and instructions for processing matrices in python
# 
# ### With Numpy that changes, when you create an array
# 
#         => it will create the elements in contiguous positions in memory
#         
#             => they will only take the amount of memory that we said they're going to take
# 
#                     => We can rely on a bunch of very efficient, low-level instructions from the CPU for matrix calculations
# 
# 
# ## Especially for Machine Learning, we need fast Array processing
# 
# ### This all applies to floats
# 
# > Completely different representation, will explore it in the course

# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>

# <h2 id="numPyArrays" style="width: 100%; height: 80px; color: rgba(255,255,255, 0.9); font-size: 30px; text-align:center; font-weight: bolder;">NumPy Arrays</h2>
# 
# 
# Major Contributions:
# 
# => Efficient numeric computation with C primitives
# 
# => Efficient collections with vectorized operations
# 
# => An integrated and natural Linear Algebra API
# 
# => A C API for connecting NumPy with libraries written in C, C++, or FORTRAN

# In[22]:


import sys

import numpy as np


# In[23]:


#Basic Arrays with Numpy

np.array([1,2,3,4])


# In[24]:


a = np.array([1,2,3,4])
b = np.array([0, .22, 1, 1.02,.5])


# In[29]:


a[0], a[1]


# In[30]:


a[0:]


# In[34]:


#steps count:
a[::2]


# In[37]:


# first option, not memory efficient!
b[0], b[2], b[-1]


# In[39]:


#multi-indexing! much easier than referencing each element alone
# this will create a numpy array => faster for processing => more memory efficient
b[[0,2,-1]]


# <img src="https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png" alt="divider" style="width: 50%; height: 15px; margin: 10px auto 0 auto; display: block"/>
# 
# ## Array Types

# In[42]:


a.dtype
# notice that numPy automatically assigned memory for it
# the numPy library needs to know the type of the object you are storing
# You cant just store a string with a number within it => it will not be able to provide performance and optimizations for arrays that are not consistent in size!


# In[43]:


# b contains floats so numPy assigns a different type for it:
b.dtype


# In[44]:


# You can change the default:

np.array([1,2,3,4], dtype=np.float)


# In[51]:


#string type
c = np.array(['a','b','c'])
c.dtype


# In[50]:


# No point of storing this in numPy, just for visualization
d = np.array([{'a':1}, sys])
d.dtype


# ## NumPy stores numbers, date, Booleans, but not a regular individual objects, as demonstrated above
# 
# ## There is a valid way to store a string, it has its own type in NumPy, but not its primary use
# 

# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Dimensions and Shapes
# 
# NumPy has several Attributes and Functions to work with multi-dimensional arrays
# 
# > shape, ndim (number of dimensions), size

# In[52]:


# this is a 2D array: 2 rows and 3 columns:

A = np.array([[1,2,3],[4,5,6]])

A.shape


# In[53]:


A.ndim


# In[54]:


A.size


# In[61]:


# Going one step further, creating B with 3D
# B in this case is a qube
B = np.array([[[12,11,10],[9,8,7]], [[6,5,4],[3,2,1]]])
print(f"<Details of B>\n\tType: {B.dtype}\n\tShape: {B.shape}\n\tNum of Dimentions: {B.ndim}\n\tSize: {B.size}")


# > If the dimensions don't match, it will return type object
# 
# > Be careful when creating by hand as dimensions must match
# 
#     example below

# In[66]:


C = np.array([[[12,11,10],[9,8,7]],[[6,5,4]]])
print(f"THIS IS HOW THE RESULT WOULD LOOK LIKE IF DIMENSIONS DONT MATCH\n\t\t\t<Details of C>\n\tType: {C.dtype}\t**Notice how the returned type is object, because of uneven dimensions (not matching)\n\tShape: {C.shape}\n\tNum of Dimentions: {C.ndim}\n\tSize: {C.size}")


# In[67]:


type(C[0])


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Indexing and Slicing of Matrices

# In[68]:


#Square matrix
A = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[77]:


print(f"\t\t\tSimilar to before, but accounting for multiple dimensions!\n=>A[1]: {A[1]}\n=>A[1][0]: {A[1][0]}\t\t**Selecting first element from second row\n=>A[1,0]: {A[1,0]}\t\t**same selection, only using multi dimensional selection of NumPy\n\t\t\t=> more memory efficient\n\t\t\t=>Remember! we are dealing on row and column levels\n=>A[:,:2]: {A[:,:2]}\t\t\t**Selecting all rows, but only first 2 columns\n=>A[:2,:2]: {A[:2,:2]}\n=>A[:2,2:]: {A[:2,2:]}\n=>A: {A}")


# In[78]:


#Re-Assigning:
A[1] = np.array([10,10,10])
A


# In[80]:


#this is an Expand operation => numPy will take care of expanding it to the selected array
A[2] = 99
A


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Summary Statistics:
# 
# There are many operations you can perform on a multi dimensional Array and matrices

# In[82]:


print(f"=> A.sum(): {A.sum()}\n=> A.mean(): {A.mean()}\n=> A.std(): {A.std()}\n=> A.var(): {A.var()}")


# ## We can do this with axis as well:
# 

# In[89]:


#sum of each column
A.sum(axis=0)


# In[90]:


#sum of each row
A.sum(axis=1)


# In[91]:


print(f"A.mean(axis=0): {A.mean(axis=0)}\nA.std(axis=0):{A.std(axis=0)}\nA.std(axis=1): {A.std(axis=1)}")


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Broadcasting and Vectorized operations:
# 
# > fundamental pieces in NumPy!
# 
# ### Vectorized operations are operations performed between both arrays and arrays, and arrays and scalars, which are optimized to be extremely fast
# 

# In[99]:


a = np.arange(4)
a


# #### "a" did not change here
# 
# #### => NumPy is an immutable first library => Any operation performed on an array will not modify it, but it will return a new array 

# In[101]:


print(f"Vectorized operations, applied on each element of the array:\n\t=> a+10 : {a+10}\n\t=> a*10: {a*10}")
a 


# #### here, the array became mutable and it changed "a", no new array returned

# In[102]:


a+=100
a


# In[103]:


l = [0,1,2,3]
#list comprehension:
[i * 10 for i in l]


# In[104]:


a = np.arange(4)
a


# In[106]:


b = np.array([10,10,10,10])
b


# In[108]:


print(f"vectorized operations can also be between arrays and arrays:\n\t=> a+b: {a+b}\n\t=> a*b: {a*b}")


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Boolean Arrays:
# 
# > Also called masks

# In[109]:


a = np.arange(4)
a


# In[112]:


#here we are selecting the first and last elements in the array, like we did before
a[[0,-1]]


# In[115]:


#I want to select these elements, in this order, will only select whats specified as True
#Usually not going to write manually true or false, this is the result of Broadcasting a boolean operation, like in the upcoming example
a[[True, False, False, False]]


# In[117]:


#Broadcasting a boolean operation
a >= 2


# In[128]:


#selecting the elements that pass (match) the boolean condition
#we are filtering/selecting the array using this boolean
a[a >= 2]


# In[120]:


a.mean()


# In[123]:


a[a>a.mean()]


# In[129]:


#all the elements that are NOT greater than the mean
a[~(a>a.mean())]


# In[125]:


a[(a==0) | (a==1)]


# In[130]:


a[(a <= 2) & (a%2==0)]


# In[131]:


A = np.random.randint(100, size=(3,3))
A


# In[132]:


A[np.array([[True,False,True],[False,True,False],[True,False,True]])]


# In[133]:


A > 30


# In[134]:


A[A>30]


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Linear Algebra:
# 
# #### NumPy already contains all the most important operations for Linear Algebra and already optimized with low level semantics => Extremely fast!

# In[144]:


A = np.array([[1,2,3],[4,5,6], [7,8,9]])
B = np.array([[6,5],[4,3],[2,1]])
#B.ndim
#A.ndim


# In[136]:


A.dot(B)


# In[137]:


A @ B


# In[138]:


B.T


# In[139]:


A


# In[140]:


B.T @ A


# #### Size of Objects in Memory
# 
# An integer in Python is larger than 24 bytes

# In[145]:


sys.getsizeof(1)


# In[146]:


#Larger numbers will even take more bytes to store them!
sys.getsizeof(10**100)


# #### NumPy size is much smaller

# In[147]:


np.dtype(int).itemsize


# In[155]:


np.dtype(np.int8).itemsize


# In[148]:


np.dtype(float).itemsize


# ### Lists are even larger in Python!

# In[149]:


#A one element list:
sys.getsizeof([1])


# ### Meanwhile in NumPy

# In[150]:


np.array([1]).nbytes


# ## Performance is also important!

# In[151]:


l = list(range(1000))


# In[152]:


a = np.arange(1000)


# In[159]:


get_ipython().run_line_magic('time', 'np.sum(a ** 2)')


# In[161]:


get_ipython().run_line_magic('time', 'sum([x ** 2 for x in l])')


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# # Pandas
# 
# > Very mature library, primary and important for data analysis and data science
# 
# 
# 
# 
# ## Step 1: Getting the data from multiple sources (Databases, Excel files, CSV files, ...)
# 
# ## Step 2: Processing the Data (combining, merging, doing different types of Analysis)
# 
# ## Step 3: Visualizing the data with Pandas
# 
# ## Step 4: Creating reports, simple statistical analysis, some ML with the help of other libraries
# 
# 

# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)

# ## Data Structures that Pandas has, how they work? How are the data structures processed?
# 
# ## 1) Pandas Series:
# 
# To demonstrate, will analyse "The Group of Seven"
# 
# > Will use pandas.Series object
# 
# > Series are very similar to numpy arrays
# 

# In[162]:


import pandas as pd
import numpy as np


# In[183]:


#In millions:
g7_pop = pd.Series([35.467,63.951,80.940,60.665,127.061, 64.511, 318.523])
g7_pop
#ordered sequence of elements, all indexed by a given index, although it looks a lot like a python list, but we there are a ton of difference


# #### Series can have a name to better document the purpose of the Series

# In[184]:


g7_pop.name = 'G7 Population in millions'
g7_pop


# In[185]:


g7_pop.dtype
#all the numbers in this object will be of type float64, and can have specific names, unlike python lists that contain many types


# In[186]:


g7_pop.values
#notice that its backed by numpy arrays!


# In[187]:


type(g7_pop.values)


# In[188]:


g7_pop


# In[189]:


g7_pop[0]


# In[193]:


g7_pop.index
#notice that series have start, stop and step index, in python list, we dont see it, in here, each element has an associated value with it in series.
#both are ordered sequence of elements, although, we can change the index of an element in a series


# In[197]:


g7_pop.index = ['Canada','France','Germany', 'Italy', 'Japan', 'United Kingdom', 'United States',]
g7_pop
#now we cna reference the series with a perticular index, that has a meaningful name, unlike python dictionaries, series are ordered!


# In[195]:


type(g7_pop.index)


# In[196]:


type(g7_pop.values)


# ### We can say that Series look like "ordered dictionaries". We can actually create Series out of dictionaries:

# In[198]:


pd.Series({'Canada': 35.467, 'France': 63.951, 'Germany': 80.940, 'Italy': 60.665, 'Japan': 127.061, 'United Kingdom': 644.511, 'United States': 318.523}, name = 'G7 Population in millions')


# In[199]:


#indexing is now done by those indices:
pd.Series(g7_pop, index=['France', 'Germany', 'Italy', 'Spain'])


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Pandas Indexing and Conditional Selection:
# 

# In[201]:


# we can consult directly
g7_pop['Canada']


# In[202]:


g7_pop['United Kingdom']


# In[207]:


#Numeric positions can also be used with the iloc attribute
g7_pop.iloc[0]


# In[206]:


g7_pop.iloc[-1]


# In[209]:


#selecting multiple elements at once:
g7_pop[['Italy', 'France']]
#The result is another Series


# In[211]:


#also sequential multi-indexing:
g7_pop.iloc[[0,1]]


# ## IMPORTANT!!
# 
# ## Series also support range or selection or slices, BUT there is a fundamental difference with Python
# 
# #### In Pyton, the upper limit of the slice is not returned
# 
# ## In Series, the upper limit is included!

# In[213]:


g7_pop['Canada':'Italy']
#Notice how Italy is included in the returned result


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Conditional Selection (Boolean Arrays):
# 
# > The same boolean array techniques that we saw applied to numpy arrays can be used for Pandas Series
# 
# ## ~ : Not
# 
# ## | : Or
# 
# ## & : And
# 
# ## All work with boolean selections
# 
# ## Boolean Series, we can perform operations on top of Series:

# In[214]:


g7_pop


# In[215]:


g7_pop > 70


# In[216]:


g7_pop.mean()


# In[222]:


g7_pop[(g7_pop > g7_pop.mean() - g7_pop.std() / 2) | (g7_pop > g7_pop.mean() + g7_pop.std() / 2)]


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Operations and Methods:
# 
# > Series also support vectorized operations and aggregation functions as NumPy

# In[218]:


g7_pop * 1_000_000


# In[219]:


g7_pop.mean()


# In[220]:


np.log(g7_pop)


# In[221]:


g7_pop['France':'Italy'].mean()


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Boolean Arrays:
# 
# > Works the same way as NumPy

# In[224]:


g7_pop[(g7_pop > 80) | (g7_pop < 40)]


# In[225]:


g7_pop[(g7_pop>80) & (g7_pop<200)]


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Modifying Series:
# 
# > Relatively simply, by index or sequential position or based on Boolean selection
# 
# ## This will be extremely important when we are cleaning data!

# In[226]:


#by index:
g7_pop['Canada'] = 40.5
g7_pop


# In[228]:


#sequential position
g7_pop.iloc[-1] = 500
g7_pop


# In[229]:


#based on boolean selection
g7_pop[g7_pop < 70] = 99.99
g7_pop


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Pandas DataFrames
# 
# > Pretty similar to excel table

# In[230]:


df = pd.DataFrame({
    'Population': [35.467,  63.951,  80.94 ,  60.665, 127.061,  64.511, 318.523],
    'GDP': [1785387, 2833687,3834343,34234432,454324,5342542,452354],
    'Surface Area': [452525,4351541,145154,4784,84676,45452,62565],
    'HDI': [0.965,0.856,0.932,0.843,0.998,0.891,0.786],
    'Continent': ['America', 'Europe','Europe','Europe', 'Asia', 'Europe', 'America']
}, columns = ['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])


# In[232]:


df
#notice how dataframes also have indexes => Pandas assigned a numeric, autoincremental index automaticcally to each "row" in our DataFrame


# In[235]:


#since in our case, we know that each row represents a country, so we can re-assign the indexes that were provided automatically:

df.index = ['Canada','France','Germany', 'Italy', 'Japan', 'United Kingdom', 'United States',]
df


# ## We can think of a dataFrame as a combination between multiple Series, 1 per column

# In[237]:


# you have several attributes that you cna consult (name of columns, index names,info(), size, shape, describe, dtypes, ...)
df.columns


# In[238]:


df.index


# In[245]:


#quick information about the structure of your dataFrame
df.info()


# In[246]:


df.size


# In[241]:


df.shape


# In[247]:


# Similar to info, to check the structure of the dataframe => summary of the statistics of the dataframe
# This is for numeric columns only => they will have summary statistics for them
df.describe()


# In[249]:


#Pandas, through NumPy, is automatically recognizing the correct type to assign to contents
df.dtypes


# In[250]:


#Quick reference of the types
df.dtypes.value_counts()


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Indexing, Selecting and Slicing DataFrames
# 
# > Individual columns in the DataFrame can be selected with regular indexing. Each column is represented as a Series:

# In[251]:


df


# In[254]:


#Selecting the row by index:
df.loc['Canada']


# In[255]:


#Selecting by Sequential position
df.iloc[-1]


# In[256]:


#if not using loc or iloc, then you are selecting the entire column, as such:
df['Population']


# ## In Summary, both loc and iloc, work in a horizontal way, selecting row, meanwhile if we dont use them and directly reference the column, then the result will be that column
# 
# ## Always what's being returned is a Series

# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Pandas Conditional Selection and Modifying DataFrames

# In[258]:


df


# In[259]:


#Selecting all countries which population is greater than 70
df['Population'] > 70


# In[260]:


df.loc[df['Population'] > 70]


# In[263]:


#specifying column as well
df.loc[df['Population'] > 70, 'Population']


# In[264]:


#specifying multiple columns:
df.loc[df['Population'] > 70, ['Population', 'GDP']]


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Dropping Stuff
# 
# > Opposite to the concept of selection
# 
# 
# ## Instead of pointing out which values you'd like to select, you could point out which ones you'd like to drop:

# In[266]:


df.drop('Canada')


# In[267]:


df.drop(['Canada', 'Japan'])


# In[268]:


#dropping columns:

df.drop(columns=['Population', 'HDI'])


# ### They have not-so recommended methods with axis, as demonstrated below

# In[269]:


df.drop(['Italy', 'Canada'], axis = 0)


# In[270]:


df.drop(['Population', 'HDI'], axis = 1)


# In[271]:


df.drop(['Population', 'HDI'], axis = 'columns')


# In[272]:


df.drop(['Canada', 'Germany'], axis = 'rows')


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Operations
# 
# ## Broadcasting operations that we are going to do between the Series

# In[273]:


df[['Population', 'GDP']]


# In[274]:


df[['Population', 'GDP']] * 100


# In[275]:


df[['Population', 'GDP']] / 100


# ## Operations with Series work at a column level, broadcasting down the rows

# In[276]:


crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])


# In[278]:


# Broadcasting Operations between Series (df and crises)
df[['GDP','HDI']] + crisis


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Modifying Data frames
# 
# > Remember, so far, we have been performing immutable operations
# 
# > The actual shape of our dataFrame and its contents are still the same
# 
# > We are not changing the existing dataFrame, rather creating a new one each one with immutable operations
# 
# > 99% of operations in Pandas is immutable, performing them will create new Series
# 
# ## How to Modify the actual contents of the dataFrame?
# 
# ## Very Simple and intuitive, You can add columns, or replace values for columns without issues

# In[279]:


#Adding new column:
langs = pd.Series(['French', 'German', 'Italian'], index = ['France', 'Germany', 'Italy'], name='Language')


# In[280]:


df['Language'] = langs


# In[281]:


df


# In[282]:


# Replacing values per column

df['Language'] = 'English'
df


# In[287]:


#Renaming Columns:
#notice these have no = sign, so they will not change the actual dataframe

df.rename(columns = {'HDI': 'Human Development Index', 'Annual Popcorn Consumption': 'APC'}, index = {'United States':'USA', 'United Kingdom':'UK','Argentina':'AR'})
#notice that values that do not exist in the df did not cause any problems
#These operations are immutable, the original df has not been changed


# In[284]:


df.rename(index=str.upper)


# In[285]:


df


# In[288]:


df.rename(index = lambda x: x.lower())


# In[289]:


df


# ## Dropping Columns:

# In[290]:


df.drop(columns = 'Language', inplace=True)


# In[291]:


df


# ## Adding Values:

# In[296]:


df.append(pd.Series({'Population':3, 'GDP':5}, name='China'))
#Append returns a new DataFrame


# In[297]:


df


# In[300]:


#we can use drop to just remove a row by index:
df.drop('Japan', inplace=True)


# In[301]:


df


# ## More Radical Index Changes

# In[302]:


df.reset_index()


# In[303]:


df


# In[304]:


df.set_index('Population')


# In[305]:


df


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Creating Columns from Other Columns

# In[308]:


df[['Population', 'GDP']]


# In[309]:


df['GDP'] / df['Population']


# In[306]:


df['GDP Per Capita'] = df['GDP'] / df['Population']


# In[307]:


df


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Reading External Data and Plotting
# 
# 
# ## Pandas can easily read data stored in different file formats like CSV, JSON, XML or even excel.
# 
# 
# ## Parsing the data always involves specifying the correct structure, encoding and other details.
# 
# ## The read_csv method reads CSV files and accepts many parameters!
# 
# > don't worry about remembering every single parameter on the top of your head, research is your best friend Checkout:
# 
# > ## => [Pandas Documentation](https://pandas.pydata.org/docs/)
# 
# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 

# In[380]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[312]:


#Notice how many parameters it accepts!
pd.read_csv


# In[324]:


#A better look at the parameters:
get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[313]:


df = pd.read_csv('btc-market-price.csv')


# In[314]:


df.head()


# 

# ## The CSV we're reading has only two columns: timestamp and price.
# 
# > ## It doesn't have a header
# 
# > ## It contains whitespace
# 
# > ## It has values seperated by commas
# 
# ## Pandas automatically assigned the first row of data as headers, which is incorrect!
# 
# ## We can overwrite this behavior with the header parameter as such:

# In[347]:


df = pd.read_csv('btc-market-price.csv', header=None)
df.head()


# ## We can then set the name of each column explicitly by setting the df.columns attribute:

# In[348]:


df.columns = ['Timestamp', 'Price']
df.head()


# In[349]:


df.shape
#365 rows & 2 columns


# In[350]:


df.info()
#notice, Price dtype is float64 which is correct, but Timestamp is showing as an object!


# In[351]:


#we can also use the df.tail() to checkout the end rows
df.tail()


# In[352]:


#specifying is possible:
df.head(10)


# In[353]:


df.tail(2)


# In[354]:


df.dtypes
#timestamp was not properly parsed as a date, it was parsed as an object


# ## We can perform a vectorized operation to parse all the Timestamp values as Datetime objects:

# In[355]:


pd.to_datetime(df['Timestamp']).head()


# In[356]:


#nothing changed because of immutability, we need to re-assign the timestamp column with the parsed values!
df.dtypes


# In[357]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.dtypes
#Now it worked!


# 

# In[358]:


df.head()


# ## The timestamp looks a lot like the index of this DataFrame
# 
# ## We can change the autoincremental ID generated by pandas and use the Timestamp DS column as the index

# In[ ]:


df.set_index('Timestamp', inplace=True)


# In[362]:


df.head()


# In[365]:


#now we can do this to get the price of Bitcoin at a certain Timestamp, because it is the index
df.loc['2017-09-29']


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Putting Everything Together
# 
# > What happens when we want to turn this into an automated script?
# 
# ## The read_csv method is very powerful, that it lets us do everything so far in just one call of the read_csv method!
# 

# In[366]:


#Here's what we did so far:

df = pd.read_csv('btc-market-price.csv', header=None)
df.columns = ['Timestamp', 'Price']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)


# In[368]:


df.head()


# ## The read_csv method is very powerful, that it lets us do everything so far in just one call of the read_csv method!

# In[373]:


df = pd.read_csv('btc-market-price.csv', header=None, names=['Timestamp','Price'], index_col=0, parse_dates=True)
df.head()
#same result, just one line of code
# we specified that the first col will be the index of the dataframe and all the rest inside the read_csv


# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# ![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)
# 
# ## Plotting Basics:
# 
# ## Pandas Integrates with Matplotlib.pyplot's interface.
# 
# ## Creating a plot is as simple as:

# In[384]:


df.plot()


# ## Behind the scenes, its using matplotlib.pyplot's inteface
# 
# ## similar plot can be created with the plt.plot() function:
# 
# > plt.plot() accepts many parameters, but the first 2 ones are the most important ones: The values of the X and Y axes

# In[385]:


plt.plot(df.index, df['Price'])


# In[387]:



x = np.arange(-10,11)
x


# In[388]:


plt.plot(x, x**2)


# ## We are using matplotlib's global API, which is horrible, but its the most popular one
# 
# ## Will learn later how to use the OOP API which will make our work much easier!

# In[389]:


plt.plot(x, x**2)
plt.plot(x, -1*(x**2))


# ## Each plt function alters the global state. If you want to set settings of your plot:
# 
# ## Use plt.figure function
# 
# ## Others like plt.title keep altering the global plot:

# In[390]:


plt.figure(figsize=(12,6))
plt.plot(x, x**2)
plt.plot(x, -1 * (x**2))

plt.title('My Nice Plot')


# ## Some of the arguments in plt.figure and plt.plot are available in the pandas' plot interface:

# In[391]:


df.plot(figsize=(16, 9), title='Bitcoin Price 2017-2018')


# ## A more Challenging parsing
# 
# lets add Ether prices to our DataFrame, it will need cleaning up

# In[396]:


eth = pd.read_csv('eth-price.csv')

print(eth.dtypes)
eth.head()


# >   ## It has a value column, which represents the price. A Date(UTC) one that has a string representing dates. And also a UnixTimeStamp date representing the datetime in unix timestamp format

# In[397]:


eth = pd.read_csv('eth-price.csv', parse_dates=True)
print(eth.dtypes)


# In[ ]:




