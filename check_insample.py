import pandas as pd

df = pd.read_csv("results/arima_insample.csv")
print(df.tail(10).to_string(index=False))
