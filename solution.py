import joblib
import pandas as pd

from features import concat_df_and_features_with_preproc

df = pd.read_csv("test.csv")
id_column = list(df['id'])

df = concat_df_and_features_with_preproc(file_path= 'test.csv', end_file_name = "test_preprocessing.csv")

model = joblib.load('RandomForestRegressor.pkl')
pred = model.predict(df)

df_submission = pd.DataFrame({"id": id_column, "score": pred})
df_submission.to_csv('submission.csv', index=False)

