def main():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib
    from sklearn.feature_extraction.text import CountVectorizer
    import streamlit as st
    import requests
    from bs4 import BeautifulSoup 
    import matplotlib.pyplot as plt

    st.title('Amazon Product Analyser')
    html_temp = """
    <div style = 'background_color:tomato ; padding;10px'>
    <h2 style = "color : white; text-align :center;">Amazon Product Analyser</h2>
    </div>
    """
    t = 0
    if t == 0:
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
        t = 1
    if t == 1:
        st.text('setup established')
        st.markdown(html_temp,unsafe_allow_html = True)
        Amazon_reviews_link = st.text_input("Review link","")
        if st.button('analyse'):
            url = 'https://www.amazon.com/product-reviews/B01J94SWWU'
            url_cut = url
            reviews_list = []
            st.text('webscraping the reviews')
            import requests
            from bs4 import BeautifulSoup

            def get_reviews(url):
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                reviews = []
                for review in soup.find_all('div', {'data-hook': 'review'}):
                    review_text = review.find('span', {'data-hook': 'review-body'}).text.strip()
                    rating = float(review.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip())
                    reviews.append(review_text)
                return reviews

                url = 'https://www.amazon.com/product-reviews/B01DFKC2SO'
                reviews = get_reviews(url)

                # fix indentation here
            st.text('webscraping completed')
            st.text('labeling good and bad started')

            sentiment = []
            for i in range(len(reviews_list)):
                sentiment.append(int(model.predict(vec.transform(reviews[i]))))

            # Pie chart, where the slices will be ordered and plotted counter-clockwise:
            labels = 'Good', 'Bad'
            sizes = [sentiment.count(0), sentiment.count(1)]
            st.text(sizes)
            
if __name__=='__main__':
    main()
