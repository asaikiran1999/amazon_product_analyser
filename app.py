
def main():
	import pandas as pd
	from sklearn.model_selection import train_test_split
	import joblib
	from sklearn.feature_extraction.text import CountVectorizer
	import streamlit as st
	import requests
	from bs4 import BeautifulSoup as bs
	import matplotlib.pyplot as plt

	st.title('Amazon Product Analyser')
	html_temp = """
	<div style = 'background_color:tomato ; padding;10px'>
	<h2 style = "color : white; text-align :center;">Amazon Product Analyser</h2>
	</div>
	"""
	t=0
	if t==0:
		st.text('setup establishment on process it takes 50 seconds')
		data = pd.read_csv("https://raw.githubusercontent.com/asaikiran1999/amazon_product_analyser/main/final_amazon_dataset.csv")
		
		# Split into training and testing data
		x = data['reviews.text']
		y = data['reviews.sentiment']
		x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

		# Vectorize text reviews to numbers
		vec = CountVectorizer(stop_words='english')
		x = vec.fit_transform(x).toarray()
		x_test = vec.transform(x_test).toarray()

		from sklearn.naive_bayes import MultinomialNB

		model = MultinomialNB()
		model.fit(x, y)
		t=1
	if t==1:
		st.text('setup established')
		st.markdown(html_temp,unsafe_allow_html = True)
		Amazon_reviews_link = st.text_input("Review link","")
		if st.button('analyse'):
			url = Amazon_reviews_link
			url_cut = url
			reviews_list = []
			st.text('webscraping the reviews')
			response = requests.get(url)
			soup = BeautifulSoup(response, 'html.parser')
			for tag in soup.find_all(['img', 'video']):
				tag.extract()
			for review in soup.find_all('div', class_='review-text-content'):
				reviews = review.get_text().strip()
			for i in range(0,len(reviews)):
				reviews_list.append(reviews[i].get_text())
			st.text('webscraping completed')
			st.text('labeling good and bad started')
			
			sentiment = []
			for i in range(len(reviews_list)):
				sentiment.append(int(model.predict(vec.transform([reviews_list[i]]))))

			
	        
			# Pie chart, where the slices will be ordered and plotted counter-clockwise:
			labels = 'Good', 'Bad'
			sizes = [sentiment.count(1),sentiment.count(0)]
			explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

			fig1, ax1 = plt.subplots()
			ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
			        shadow=True, startangle=90)
			ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

			st.pyplot(fig1)
if __name__=='__main__':
	main()
