import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

# Plot class distribution
sns.countplot(x="label", data=df, palette="coolwarm")
plt.title("Spam vs. Non-Spam SMS Distribution")
plt.xlabel("Message Type (0 = Non-Spam, 1 = Spam)")
plt.ylabel("Count")
plt.savefig("reports/class_distribution.png")  # Save plot
plt.show()
