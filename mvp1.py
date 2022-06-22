from sklearn.linear_model import Ridge
import streamlit as st
import pandas as pd
import sklearn_json as skljson

#!pip install lime
import lime
import lime.lime_tabular
import numpy as np


diamonds = pd.read_csv("diamonds.csv", sep=";", decimal=".")
feedback = pd.read_csv("feedback.csv", sep=";")

#!pip install sklearn-json

#Loading up the Regression model we created
model = Ridge()
model = skljson.from_json('rr_model.json')



estimator = model
x_featurenames = diamonds.columns

explainer1 = lime.lime_tabular.LimeTabularExplainer(np.array(diamonds),
                    feature_names=x_featurenames, 
                    #class_names=['pIC50'], 
                    # categorical_features=, 
                    # There is no categorical features in this example, otherwise specify them.                               
                    verbose=False, mode='regression')


#Caching the model for faster loading
@st.cache



def predict(carat, cut, color, clarity, depth, table, x, y, z):
    #Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6
    
    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7
    

    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']))
    return prediction

#st.title('Diamond Price Predictor')
#st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.image("handLens.PNG")
st.subheader('Explanation setup:')

n = st.number_input('Number of features:', min_value=2, max_value=8, value=2)

i = st.number_input('Instance ID:', min_value=0, max_value=100, value=0)


if st.button('Show explanation'):
    explanation = explainer1.explain_instance(diamonds.iloc[i,:], estimator.predict, num_features=n)
    #price = predict(carat, cut, color, clarity, depth, table, x, y, z)
    #st.success(f'The predicted price of the diamond is ${price[0]:.2f} USD')
    st.pyplot(explanation.as_pyplot_figure())
    txt = st.text_area('Feedback')
    if st.button('Submit'):
      st.session_state.feedback = st.session_state.feedback.append({'id': i, 'feedback': txt}, ignore_index=True)

      #element = st.dataframe(feedback)
      #data = [[i, txt]]
      #df = pd.DataFrame(data, columns=['id', 'feedback'])
      #element.add_rows(df)
      #st.session_state['feedback'] = st.session_state['feedback'].append(data, ignore_index=True)
      #st.dataframe(st.session_state['feedback'])
      #feedback = feedback.append({'id': i, 'feedback': txt}, ignore_index=True)
      #feedback.to_csv("feedback.csv", index=False, header=True)
      #st.session_state['feedback'] = st.session_state['feedback'].append(data, ignore_index=True)
      #st.dataframe(st.session_state['feedback'])
      #st.dataframe(st.session_state.feedback)
      #st.write(feedback)
      
