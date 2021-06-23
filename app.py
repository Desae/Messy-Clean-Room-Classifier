import streamlit as st
from PIL import Image
from img_classification import machine_classification

st.title("Image Classification")
st.header("Messy/Clean Room Image Classifier")
st.text("Upload a room image for image classification as clean or messy")

uploaded_file = st.file_uploader("Choose a room image ...", type=["png","jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', width=300)
    st.write("")
    st.write("Classifying...")
    label = machine_classification(image, 'messy_clean_room2.h5')
    st.write("This room is ", label['status'], " with a probability of ", label['prob'].item())
    st.slider("Degree of Cleanliness/Degree of Messiness", value=label['prob'].item(),
    min_value=0.0, max_value=1.0, step=0.001)
