import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction

def model_prediction(test_image):
    cnn = tf.keras.models.load_model('trained_flowers_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = cnn.predict(input_arr)
    result_index = np.argmax(prediction) #Return index of max element
    return result_index
 
#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Flower Species Prediction"])

#Home Page
if (app_mode == "Home"):
    st.header("FLOWER SPECIES PREDICTION SYSTEM")
    image_path = "home_page_image.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown(""" 
    Welcome to the Flower Species Prediction System!
    This system merges botanical science with AI, using advanced algorithms to accurately identify and predict flower species.
     It is user-friendly and can be used by botanists, horticulturists and enthusiasts to explore floral diversity.    

    ### How It Works
    1. **Upload Image:** Go to the **Flower Species Prediction** page and upload an image of a flower to be predicted.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify the species of the flower.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for the prediciton of flower species.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds.

    ### Get Started
    Click on the **Flower Species Prediction** page in the sidebar to upload an image and experience the power of our Flower Species Prediction System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                       
    """)
#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset was downloaded from Kaggle website using this link: https://www.kaggle.com/datasets/junkal/flowerdatasets
    This dataset consists of about 11,531 images of different species of flowers which is categorized into 7 different classes.The total dataset is divided into 70/20/10 ratio of training, validation and test sets preserving the directory structure.
    .
    #### Content
    1. train (8069 images)
    2. test (1156 images)
    3. validation (2306 images)

    """)

#Prediction Page
elif(app_mode=="Flower Species Prediction"):
    st.header("Flower Species Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name =['daisy', 'dandelion', 'lily', 'orchid', 'rose', 'sunflower', 'tulip']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))