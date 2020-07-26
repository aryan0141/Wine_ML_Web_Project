from django.http import HttpResponse
from django.shortcuts import render, redirect
import pandas as pd                                                  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


wine=pd.read_csv("data.csv")
train_set, test_set  = train_test_split(wine, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(wine, wine['Alchohol']):
	strat_train_set = wine.loc[train_index]
	strat_test_set = wine.loc[test_index]
wine = strat_train_set.copy()
	
wine = strat_train_set.drop("Wine", axis=1)
wine_labels = strat_train_set["Wine"].copy()
#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import Perceptron
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors=3)
#X, y = load_digits(return_X_y=True)
#model = Perceptron(tol=1e-3, random_state=0)
#model= GaussianNB()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(wine, wine_labels)

test_features=strat_test_set.drop("Wine", axis=1)
test_labels=strat_test_set["Wine"].copy()

y_labels=model.predict(test_features)

x=list(y_labels)
y=list(test_labels)
accuracy=[]
for i in range(len(test_labels)):
	if x[i]>y[i]:
		accuracy.append((y[i]/x[i])*100)
	else:
		accuracy.append((x[i]/y[i])*100)

acc=sum(accuracy)/len(x)

def index(request):
	if request.method == 'POST': 
		alchohol_content=request.POST.get('alchohol_content','default')
		malic_acid=request.POST.get('malic_acid','default')
		Ash=request.POST.get('Ash','default')
		alc_ash=request.POST.get('alc_ash','default')
		Magnesium=request.POST.get('Magnesium','default')
		Phenols=request.POST.get('Phenols','default')
		Flavanoid=request.POST.get('Flavanoid','default')
		NFPhelons=request.POST.get('NFPhelons','default')
		Cyacnins=request.POST.get('Cyacnins','default')		
		Intensity=request.POST.get('Intensity','default')
		Hue=request.POST.get('Hue','default')
		OD280=request.POST.get('OD280','default')
		Proline=request.POST.get('Proline','default')

		labels=[[float(alchohol_content),
			float(malic_acid),
			float(Ash),
			float(alc_ash),
			float(Magnesium),
			float(Phenols),
			float(Flavanoid),
			float(NFPhelons),
			float(Cyacnins),
			float(Intensity),
			float(Hue),
			float(OD280),
			float(Proline)
		]]

		our_labels = model.predict(labels)

	
		

		if our_labels[0]<=400:
			wine_quality="A Poor Quality Wine"
		if 400<our_labels[0]<=800:
			wine_quality="A Average Quality Wine"
		if 800<our_labels[0]<=1200:
			wine_quality="A Good Quality Wine"
		if 1200<our_labels[0]<=1500:
			wine_quality="A Exclusive Wine"
		if our_labels[0]>1500:
			wine_quality="A Premium & Fresh Wine"		

		details={
			"answer":our_labels[0],
			"wine_quality":wine_quality,
			"accuracy":"{:.2f}".format(acc)
		}	

	
		return render(request,"success.html",details)


	return render(request,"index.html")
	