
import streamlit as st

import pickle
import numpy as np
model1=pickle.load(open(r'C:\Users\Shiva\PycharmProjects\working nlp\model1.pkl','rb'))

def predict_NEWS(text):
    prediction=model1.predict([text])
    pred=prediction[0]
    return (pred)

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">TRUE AND FAKE NEWS DETECTION </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    text = st.text_input("TEXT","Type Here")


    if st.button("Predict"):
        output=predict_NEWS(text)
        output=output.upper()
        st.success('THE NEWS IS {}'.format(output))


if __name__=='__main__':
    main()

