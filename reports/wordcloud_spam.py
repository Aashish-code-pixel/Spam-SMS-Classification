from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Filter spam messages
spam_text = " ".join(df[df["label"] == "spam"]["cleaned_text"])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap="Reds").generate(spam_text)

# Save and display
wordcloud.to_file("reports/wordcloud_spam.png")
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
