import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


data = pd.read_excel("./Student_Mental_Health_Cleaned.xlsx")

text_data = ' '.join(data['Course'].dropna().astype(str))
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text_data)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Course Names")
plt.show()

