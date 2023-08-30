import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""In the ml, we cannot work with many datasets as it will get difficult.
We need to club all datasets into 1 dataset and work in it."""

# NOTE: whenever there is no. before \ , then we need to keep \\ in the file path to not get errors.

t = pd.read_csv("D:\data_science\project_2\train_walmart.csv")   
s = pd.read_csv("D:\data_science\project_2\stores.csv")
f = pd.read_csv("D:\data_science\project_2\features.csv")


"""Now, let's identify the NaN values in each datasets"""

print(t.isnull().sum())
print(s.isnull().sum())
print(f.isnull().sum())

print(f.shape)

# lets, see how much percent of data is missing in each columns.

print(f["MarkDown1"].isnull().sum()/f.shape[0]*100)
print(f["MarkDown2"].isnull().sum()/f.shape[0]*100)
print(f["MarkDown3"].isnull().sum()/f.shape[0]*100)
print(f["MarkDown4"].isnull().sum()/f.shape[0]*100)
print(f["MarkDown5"].isnull().sum()/f.shape[0]*100)
print(f["CPI"].isnull().sum()/f.shape[0]*100)
print(f["Unemployment"].isnull().sum()/f.shape[0]*100)

# more than 50% lost data in markdowns.. but CPI and Unemployment has very less NaN 


"""So, lets forwardfill CPI and Unemloyment and drop whole MarkDowns.. column """

f.drop(["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"],axis=1,inplace=True)

f["CPI"].fillna(method="ffill",inplace=True)
f["Unemployment"].fillna(method="ffill",inplace=True)

#[OR]

f.fill(method="ffill",inplace=True)       # as other than CPI and Unemployment remaining all the columns has no NaN values.

print(f.isnull().sum())


"""Now, we need to merge the datasets into 1 dataset by using joints.
1) Inner join --> pd.merge(a,b,on=["key_taken/common_Column"],how="inner")
2) left join --> pd.merge(a,b,on=["key_taken/common_Column"],how="left")
3) right join --> pd.merge(a,b,on=["key_taken/common_Column"],how="right")
4) outer join/full join --> pd.merge(a,b,on=["key_taken/common_Column"],how="outer")

Here, merge() cann be used as function (pd.merge(xxx)) or method(x.merge(xxx)) as well
eg: t.merge(s,how="inner",on=["Store"])

on= --> common_column taken as index/reference

**NOTE: Using of .merge() method is recommended..**"""

a = pd.merge(t,s,on="Store",how="inner")
print(a.shape)            # gives shape of dataset which helps us to know any dropped rows..

print(a.columns)          # **tells us the common columns of dataset "a" and "f" which can be used as common columns in "on=" -- > 
                          # helps in not to create duplicate columns in the dataset..**

a = a.merge(f,on=["Store","IsHoliday","Date"],how="inner")

# info of the each column of new dataframe
print(a.columns)
print(a.info())
print(a.describe())


"""This new Dataset/DataFrame does not has NaN values as we already filtered it before indivisually..
Q) covert date column into datetime and create new week and year column"""

a["Date"] = pd.to_datetime(a["Date"],errors="coerce")    # Here, we did not use "format=" because it is already in good format. And we need not necessary use "errors=" because there is not other signs in the datetime.

# **In the previous Project, we used DatetimeIndex to find month. And that method is used after converting "datetime" format --> "DatetimeIndex" format only
# and only month, weekday, etc.. will be get. To get year,week and other useful things, we can directly use the "Datetime" format and use like in the below method..**

# NOTE: below method is recomended..

a["WEEK"] = a.Date.dt.isocalendar().week    # From "dt" datetime module (a["Date"] --> Datetime format) , we oppted isocalendar() fnction and from that we find the week.
a["YEAR"] = a.Date.dt.isocalendar().year    # From "dt" datetime module (a["Date"] --> Datetime format) , we oppted isocalendar() fnction and from that we find the year.


"""Q) create a function that takes feature name as input and create scatter plot of given feature and weekly sales"""

# NOTE: recomended to use seaborn plots as they have built in xlabel, ylabel etc and more convient..

# Test
sns.scatterplot(x=a["CPI"],y=a["Weekly_Sales"],marker="*",data=a)   # sns.scatterplot(x,y,size="size_of_marker",marker="marker_type",data=) . Default marker="*"


def scattplt(x):
    sns.scatterplot(x=x,y=a["Weekly_Sales"],size=0.01,data=a)
    #[or]
#def scattplt(x):    
    #sns.scatterplot(x=a[x],y=a["Weekly_Sales"],size=0.01,date=a)
    
    
# Result    
scattplt(a["Temperature"])

# remodeling
plt.subplot(2,2,1)
scattplt(a["Temperature"])
plt.subplot(2,2,2)
scattplt(a["Fuel_Price"])
plt.subplot(2,2,3)
scattplt(a["CPI"])
plt.subplot(2,2,4)
scattplt(a["Unemployment"])

scattplt(a["Store"])
scattplt(a["Size"])
scattplt(a["Type"])


"""Q) create line chart for week_sales vs each year
Q) avg weekly sales for 2012"""

a[a["YEAR"] == 2012].groupby(["WEEK"])["Weekly_Sales"].mean().plot(kind="line")     # filtered data of only 2012 year --> gets groupedby according to each "WEEK"'s weekly_Sales's mean/avg..

# [or] --> can use seaborn as: RECOMMENDED
# sns.lineplot(x=a[a["YEAR"] == 2012].groupby(["WEEK"])["Weekly_Sales"].mean().index,y=a[a["YEAR"] == 2012].groupby(a["WEEK"])["Weekly_Sales"].mean().values,color="red")

# NOTE: Always check and read the plot/function/method Arguments INFO when we get error.

def yearplt(p):
    a[a["YEAR"] == p].groupby(a["WEEK"])["Weekly_Sales"].mean().plot(kind="line")
    
plt.figure(figsize=(15,30))    
plt.subplot(3,1,1)
yearplt(2010)
plt.title("2010 YEAR")
plt.subplot(3,1,2)
yearplt(2011)
plt.title("2011 YEAR")
plt.subplot(3,1,3)
yearplt(2012)
plt.title("2012 YEAR")

print(a.info())
print(a.describe())

# In above it gives "WEEK" & "YEAR" as UInt32. lets convert it to "float" daatatype for smooth execution.  --> this solves "sns.lineplot" error above..
a["WEEK"] = a["WEEK"].astype(float)
a["YEAR"] = a["YEAR"].astype(float)

print(a.info())

# Now, lets create all lineplots in 1 plot

plt.figure(figsize=(20,10))
yearplt(2010)
yearplt(2011)
yearplt(2012)
plt.title("WEEKLY_SALES")
plt.xlabel("WEEK")
plt.ylabel("SALES")
plt.grid()
plt.xticks(np.arange(1,60))                       # gives the "xticks" values. Can also rotate the "xticks". Here, "xticks" --> "xlabels"/"xcoordinate"
plt.legend([2010,2011,2012],loc="upper left")         # syntax --> plt.legend([order_wise_line_labels],loc="location_of_legend")

# it is seen that everything is same in every year in the plot. As 2012 plot date is lost at the end
# Now, we need to assume that we need to have high stocks as sales are getting high in the end of each year.

"""Now, lets see the distroibution of the Weekly_Sales"""

sns.distplot(a["Weekly_Sales"],kde=False,bins=100,color="red")

"""Q) plot the stores with highest avg sales
Q) plot the dept with with highest avg sales"""

plt.figure(figsize=(30,10))
a.groupby(["Store"])["Weekly_Sales"].mean().plot(kind="line")
a.groupby(["Dept"])["Weekly_Sales"].mean().plot(kind="line")

# BAR: "Store"

plt.figure(figsize=(15,9))
sns.barplot(x=w.index,y=w.values.reshape(-1,),palette="dark")   # got data structure error  [Must be in set_index("Store")]
plt.title("WEEKLY_STORE_SALES")
plt.xlabel("STORE")
plt.ylabel("WEEKLY_SALES")
plt.grid()

# check the shape
w.values   # it is in DataFrame
w.values.reshape(-1,)      # converted to 1 dimension (can say list/ndarray) to 

# [OR]

w = a.groupby(["Store"])["Weekly_Sales"].mean().reset_index()      # .reset_index() --> makes it to the default index, instead of "Store"(A particular data column) as the index.

# Now, lets see a other type of bar graph

w.sort_values("Weekly_Sales").style.bar(align="left")    # sorts values(not indices which we resetted by reset_index) according to "Weekly_Sales" ascending order & styled into bar graph on the left.

# If you want only "Weekly_Sales" column just do:
w.sort_values("Weekly_Sales")[["Weekly_Sales"]].style.bar(align="left")   # align= --> allignment of the given graph

# or we can just declare "Store" as index for w and do sort:

w = w.set_index("Store")         # directly setting "Store" as index instead of reset_index() indices which can make overall 2 columns
w.sort_values("Weekly_Sales").style.bar(align="left")

# BAR: "Dept"

d = a.groupby(["Dept"])["Weekly_Sales"].mean().reset_index()
d = d.set_index("Dept")

d.sort_values("Weekly_Sales").style.bar(align="left")

# If you want both columns

d = d.reset_index()
d.sort_values("Weekly_Sales").style.bar(align="left")

#[OR]

plt.figure(figsize=(15,9))
sns.barplot(x=d.index,y=d.values.reshape(-1,),palette="dark")   # got data structure error  [Must be in set_index("Dept")]
plt.title("WEEKLY_DEPT_SALES")
plt.xlabel("DEPT")
plt.ylabel("WEEKLY_SALES")
plt.grid()

# check the shape
d.values   # it is in DataFrame
d.values.reshape(-1,)      # converted to 1 dimension (can say list/ndarray) to 


"""Lets see the heatmap of the data"""

plt.figure(figsize=(15,20))
sns.heatmap(a[["Store","Dept","Weekly_Sales","IsHoliday","Size","Temperature","Fuel_Price","CPI","Unemployment","WEEK","YEAR"]].corr(),annot=True)

a.info()


"""Lests see the relation between type vs size"""

a.groupby(["Type"])["Weekly_Sales"].mean().plot(kind="bar")

sns.boxplot(x=a["Type"],y=a["Size"],data=a)


"""Lets see year vs unemployment"""

a.groupby(["YEAR"])["Unemployment"].sum().plot(kind="bar")


"""lets see year vs holidays"""


a.groupby(["YEAR"])["IsHoliday"].sum().plot(kind="bar")


"""lets see the temperature"""

a["Temperature"].value_counts()           # gives no. of times the temp(not repeating) are repeating


"""REPORT:
    1) dept 1 is highest unemployment
    2) dept 43 is lowest unemployment
    3) type c stores are smaller in size, type a is larger in size
    4) type a has high avg sales, type c has low avg sales  
    5) store 20 has highes sales
    6) store 5 has lowest sales
    7) every year sales are pretty same and end of year is unexpectedly high
    8) dept 1 has highest avg sales
    9) high correlation b/w year and fuel price
    """
    
    
    
    
    
    
    
    
    
