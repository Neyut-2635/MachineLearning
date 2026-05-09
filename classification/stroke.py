import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('/Users/thanhtuyen2603/Documents/ML/MachineLearning/datasets/stroke_classification.csv')

profile = ProfileReport(df, title='STROKE', explorative=True)
profile.to_file('stroke_report.html')

X = df.drop([''])