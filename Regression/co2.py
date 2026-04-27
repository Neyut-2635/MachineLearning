import math
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv('../datasets/co2.csv')
profile = ProfileReport(data, title="CO2 Emissions Report", explorative=True)
profile.to_file('../reports/co2.html')

target = "CO2 Emissions(g/km)"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer =Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy='median')),
    ('scaler', StandardScaler())
])

norm_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ('transform', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num_transformer', num_transformer, [
        'Engine Size(L)',
        'Fuel Consumption City (L/100 km)',
        'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (L/100 km)',
    ]),
    ('norm_transformer', norm_transformer, [
        'Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'
    ])
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
for predict, actual in zip(y_pred, y_test):
    print(f'predicted: {predict: .2f}, actual: {actual}')

print(f'loss: {math.sqrt(mean_squared_error(y_test, y_pred)): .2f}')