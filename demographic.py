#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:06:24 2022

@author: leechenhsin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  

# load the dataset
Demogra1 = pd.read_csv("/Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/Detail Active Report with Demographics 10.2022 _new column.csv")
Demogra1 

#drop ethnicity

Demogra1= Demogra1[Demogra1['Ethnicity'].notna()]
Demogra1= Demogra1[Demogra1['Manager Flag'].notna()]


# replacing values

Demogra1['Ethnicity'] = pd.factorize(Demogra1.Ethnicity)[0]
Demogra1['Person Gender'] = pd.factorize(Demogra1["Person Gender"])[0]
Demogra1['Job Function Name'] = pd.factorize(Demogra1['Job Function Name'])[0]
Demogra1['Years of Service'] = pd.factorize(Demogra1['Years of Service'])[0]


#prediction model _01
x1 = Demogra1 [['Ethnicity','Person Gender','Age']]
y1 = Demogra1 [['Manager Flag']]
print(x1)
print(y1)

np.random.seed(seed=13579)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2,random_state=13579)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x1_train, y1_train)

y1_pred  =  classifier.predict(x1_test)
y1_pred  


y1_test


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y1_test, y1_pred)
ac = accuracy_score(y1_test,y1_pred)

cm
ac

#prediction model _02


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# 查看數據的變數、總數、缺失數據、變數measurement(維度)
def data_overview():
    print("Rows :  " , Demogra1 .shape[0])
    print("Columns:  " , Demogra1 .shape[1] )
    print('Missing Value number : ' , Demogra1 .isnull().sum().values.sum()) #isnull.sum()會對每條series做sum up ，所以我們還要取出value做一次sum up .
    print('\nUnique values' , Demogra1 .nunique())
data_overview()



# 作圖發現有極端值的嫌疑
sns.distplot(Demogra1.Age)





x2 = Demogra1 [['Ethnicity','Age','Years of Service']]
y2 = Demogra1 [['Manager Flag']]
print(x2)
print(y2)

np.random.seed(seed=13579)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2,random_state=13579)

x2_test



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x2_train = sc.fit_transform(x2_train)
x2_test = sc.transform(x2_test)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x2_train, y2_train)

y2_pred  =  classifier.predict(x2_test)
y2_pred  




from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y2_test, y2_pred)
ac = accuracy_score(y2_test,y2_pred)


cm
ac

pip install seaborn
import seaborn as sns

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y2_test, y2_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('confusion_matrix for manager')




#test output yes/no>> tableau find insights(test set 20%)


x3 = Demogra1 [:]
y3 = Demogra1 [['Manager Flag']]

from sklearn.model_selection import train_test_split
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2,random_state=13579)

x3_test
x3_test.reset_index()
y2_pred


y2_pred_df = pd.DataFrame(y2_pred, columns = ['manager_pred'])
y2_pred_df

compression_opts = dict(method='zip',
                        archive_name='out.csv')  
y2_pred_df.to_csv('out.zip', index=False,
          compression=compression_opts)  

prediction=pd.concat([x3_test,y2_pred_df], axis=1)
prediction

import os  
os.makedirs('Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589', exist_ok=True)  
prediction.to_csv('Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/prediction.csv')



from pathlib import Path  
filepath = Path('Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/x3_test.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
prediction.to_csv(filepath)  

from pathlib import Path  
filepath = Path('Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/y2_pred_df.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
prediction.to_csv(filepath) 




#impact: 給一個model找出離升遷機會高的人有誰 找報告來support 建議下一步～ 





Demogra1 ['Person Gender'].replace(['Female', 'Male'],
                        [0, 1], inplace=True)




Demogra1.loc[Demogra1['shield'] > 6]


x = marketing_df [['age','duration','campaign']]
y = marketing_df[['y']]
print(x)
print(y)


np.random.seed(seed=13579)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=13579)

