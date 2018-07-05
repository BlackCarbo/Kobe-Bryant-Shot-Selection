#kobe_dataset
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

x = pd.read_csv('kobe_data.csv',usecols=['loc_x','loc_y','shot_distance',
'shot_zone_area','playoffs','period','minutes_remaining','seconds_remaining',
'shot_type'])

x = x.values
y = pd.read_csv('kobe_data.csv',usecols=['action_type'])
y = y.values

#print(x_data)
#print(y_data)
x_data = x[1001:30000,:]
y_data = y[1001:30000,:]
x_test = x[0:1000,:]
y_test = y[0:1000,:]
#print(y_test)

model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)
model.fit(x_data,y_data)
print(model)

expected = y_test
predicted = model.predict(x_test)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
