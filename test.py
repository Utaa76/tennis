import glob
import pandas as pd

csv_files = glob.glob("data/*.csv")  # ou ton chemin exact
df_list = [pd.read_csv(file, sep=";", encoding="latin1") for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

df_train = df[df['Date'] < '2025-01-01']
df_test = df[df['Date'] >= '2025-01-01']