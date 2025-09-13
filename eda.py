import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv("titanic.csv")
print(df)

msno.bar(df,figsize=(10,5),color="tomato")
plt.title("Bar plot showing missing data values",size = 15, c="r")
plt.show()

df.drop(['Cabin'],axis=1,inplace=True)
print(df.shape)

print(df["Embarked"].unique())
print(df["Embarked"].value_counts())
df["Embarked"] = df["Embarked"].fillna("S")
print(df["Embarked"].value_counts())
print(df.describe().astype(int))
df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df.describe().astype(int))
print(df['Age'].isnull().sum())

msno.bar(df,figsize=(10,5),color="tomato")
plt.title("Bar plot showing missing data values",size = 15, c="r")
plt.show()

sns.countplot(x="Survived",data=df)
plt.title("count of passengers who survived")
plt.show()

fig,axes = plt.subplots(1,2,figsize=(5,3))
df["Sex"].value_counts().plot(kind="bar",ax=axes[0],color=["DarkRed","Indianred"])
df["Sex"].value_counts().plot(kind="pie",ax=axes[1],autopct = '%1.0f',colormap="Reds")
plt.show()

sns.catplot(x="Sex",hue="Survived",kind="count",data=df,height=3)
plt.show()

sns.catplot(x="Pclass",hue="Survived",kind="count",data=df,height=3)
plt.show()

df.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
print(df.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])
print(df.corr())

df["Age"] = df["Age"].replace(np.nan, 0)
df["Embarked"] = df["Embarked"].replace(np.nan, 0)
print(df)

x = df.drop(["Survived"],axis=1)
y = df["Survived"]
print("XXXX",x)
print("YYYY",y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('df',df.shape)
print('x_train',x_train.shape)
print('x_test',x_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)

y_pred = NB.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)

from sklearn.metrics import accuracy_score
print('ACCURACY is', accuracy_score(y_test,y_pred))



