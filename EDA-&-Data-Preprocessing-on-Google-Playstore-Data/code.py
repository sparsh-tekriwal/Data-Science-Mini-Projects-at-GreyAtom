# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(path)
#data["Rating"].dropna(inplace=True)
plt.hist(data["Rating"], bins=50, range=[0,5.5])

data= data[data["Rating"]<=5]
plt.hist(data["Rating"])
#Code starts here


#Code ends here


# --------------
# code starts here

total_null= data.isnull().sum()
print(total_null)

percent_null= total_null/len(data)
print(percent_null)

missing_data= pd.concat([total_null, percent_null], axis=1,keys= ["Total", "Percent"])

print(missing_data)

#drop

data.dropna(inplace=True)
total_null_1= data.isnull().sum()
percent_null_1=total_null_1/len(data)
missing_data_1= pd.concat([total_null_1, percent_null_1], axis=1,keys= ["Total", "Percent"])
print(missing_data_1)

# code ends here


# --------------

#Code starts here

sns.catplot(x="Category",y="Rating",data=data, kind="box", height= 10)


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

print(data.Installs.value_counts())

data.Installs= data.Installs.str.replace(",", "").str.replace("+","")
data.Installs=data.Installs.astype(int)
print(data.Installs.value_counts())

le=LabelEncoder()
data.Installs= le.fit_transform(data.Installs)

reg= sns.regplot(x="Installs", data=data, y="Rating")
reg.set_title("Rating vs Installs [RegPlot]")

#Code ends here



# --------------
#Code starts here

#print(data.Price.value_counts())

data.Price=data.Price.str.replace("$","")
data.Price=data.Price.astype(float)
reg=sns.regplot(x="Price", y="Rating", data=data)


#Code ends here


# --------------

#Code starts here

#print((data.Genres.unique()))
data.Genres= data.Genres.str.split(";",n=1, expand = True)[0]
print(data.Genres)
gr_mean=data[["Genres","Rating"]].groupby(["Genres"], as_index=False).mean()
print(gr_mean)
print(gr_mean.describe())
gr_mean=gr_mean.sort_values(by=["Rating"])

#Code ends here


# --------------

#Code starts here

data["Last Updated"]=pd.to_datetime(data["Last Updated"])
print(data["Last Updated"])
max_date= max(data["Last Updated"])
print(max_date)

data["Last Updated Days"]= (max_date-data["Last Updated"]).dt.days
print(data["Last Updated Days"])

reg= sns.regplot(x="Last Updated Days", y="Rating", data=data)
#Code ends here


