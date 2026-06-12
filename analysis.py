import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("fear_greed_index.csv")

# Basic information
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nSentiment Counts:")
print(df["classification"].value_counts())

# Sentiment Distribution Chart
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="classification",
    order=df["classification"].value_counts().index
)

plt.title("Distribution of Fear & Greed Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("sentiment_distribution.png")

plt.show()

print("\nChart saved as sentiment_distribution.png")
