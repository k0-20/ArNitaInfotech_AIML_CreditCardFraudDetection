import pandas as pd

# Load dataset
df = pd.read_csv("data/fraudTrain.csv")

print("="*60)
print("FIRST 5 ROWS")
print("="*60)
print(df.head())
import pandas as pd

# Load dataset
df = pd.read_csv("data/fraudTrain.csv")

print("="*60)
print("FIRST 5 ROWS")
print("="*60)
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nFraud Distribution:")
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nFraud Distribution:")
print(df["is_fraud"].value_counts())