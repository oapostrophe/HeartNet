from fastai.vision.all import *
from io import BytesIO
import requests
import streamlit as st

"""
# HeartNet
This is a classifier for images of 12-lead EKGs.  It will attempt to detect whether the EKG indicates an acute MI.  It was trained on simulated images.
"""

def predict(img):
    st.image(img, caption="Your image", use_column_width=True)
    pred, _, probs = learn_inf.predict(img)
    # st.write(learn_inf.predict(img))

    f"""
    ## This **{'is ' if pred == 'mi' else 'is not'}** an MI (heart attack).
    ### Probability of MI: {probs[0].item()*100: .2f}%
    ### Probability Normal: {probs[1].item()*100: .2f}%
    """


path = "./"
learn_inf = load_learner(path + "demo_model.pkl")

option = st.radio("", ["Upload Image", "Image URL"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Please upload an image.")

    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)

else:
    url = st.text_input("Please input a url.")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PILImage.create(BytesIO(response.content))
            predict(pil_img)

        except:
            st.text("Problem reading image from", url)