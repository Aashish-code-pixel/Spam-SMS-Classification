# Filter non-spam messages
ham_text = " ".join(df[df["label"] == "ham"]["cleaned_text"])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap="Blues").generate(ham_text)

# Save and display
wordcloud.to_file("reports/wordcloud_ham.png")
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
