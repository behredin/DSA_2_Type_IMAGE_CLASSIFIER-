import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
### load file
def creds_entered():
    if st.session_state["user"].strip() =="behre" and st.session_state["passwd"].strip() =="behre":
       st.session_state["authenticated"]=True
    else:
        st.session_state["authenticated"]=False
        st.error("Invalid username or password :face_with_raised_eyebrow:")
def authenticate_user():
    if "authenticated" not in st.session_state:
            st.text_input(label="UserName: ",value="",key="user",on_change=creds_entered)
            st.text_input(label="Password :",value="",key="passwd",type="password",on_change=creds_entered)
            return False
    else:
            if st.session_state["authenticated"]:
                return True
            else:
                    st.text_input(label="UserName: ",value="",key="user",on_change=creds_entered)
                    st.text_input(label="Password :",value="",key="passwd",type="password",on_change=creds_entered)
                    return False
if authenticate_user():
    with st.sidebar:
        selected = option_menu(menu_title="Main Menu", options=['Home','Classification','About'],
                                icons=['house','book','envelope'],orientation="Horizontal",
        styles={
        "container":{"background-color":"#EC7063"},
            "nav-link":{"font-size":"21px","--hover-color":"#C843335","color":"#17202A" },
            "nav-link-selected":{"background-color":"#F8F521"},
            "icon":{"font-size":"20px" }}, )
    if(selected=="Home"):
        st.markdown("""
        <style>
            .big-foot1{font-size:50px|important;color:green;text-align:center;
            font-weight:bold;background-color:yellow;}</style>
        """,unsafe_allow_html=True)
        st.markdown('<h1 class="big-foot1">Fundamentals of Data Science and Analytics Project</h1>',unsafe_allow_html=True)
        st.markdown("""
        <style>
        .paragraph{font-size:20px |important;text-align:justify;}</style>
        """,unsafe_allow_html=True)
        st.markdown('<h3 style=color:red>Hello  Welcome to My Image Classification Dashboard App </h3>',unsafe_allow_html=True)
        st.subheader("To use this streamlit app for image classification please click Classification button from main menu ")
        st.markdown('<h1 style="text-align:center;font-weight: bold; background-color:green">እንኳን ደህና መጡ። ይህ የImage Classification Dashboard ነው። </p>',unsafe_allow_html=True)
    if(selected=='Classification'):
        def load_model():
            model = tf.keras.models.load_model("F:/New chapter/PG/DSA/Behredin_Image_CLa/behremodel2.hdf5")
            return model
        with st.spinner('Model Behre is loading Please wait..'):
            model=load_model()
        st.markdown("<h3 style=color:green> Hello!! welcome this is Image Classifier Dashboard Browse Your Image and Click Predict Button below</>",unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg","PNG","JPEG"])

        map_dict = {0: 'Fears',
                    1: 'Shame'

                    }
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(opencv_image,(224,224))
            # Now do something with the image! For example, let's display it:
            st.image(opencv_image, channels="RGB")

            resized = mobilenet_v2_preprocess_input(resized)
            img_reshape = resized[np.newaxis,...]

            Genrate_pred = st.button("Predict")    
            if Genrate_pred:
                prediction = model.predict(img_reshape).argmax()
                st.title("Ohhh...This Guy  Looks Like  {}".format(map_dict [prediction]))
    if(selected=="About"):
        st.markdown("""
        <style>
        .paragraph{
            font-size:20px |important;text-align:justify;
        }
        </style>
        """,unsafe_allow_html=True)
        st.markdown('<p class="paragraph">One of our goal is Creating Multiple Image Classification App that classifies a given image Either fear or Shame. </p> ',unsafe_allow_html=True)
        st.markdown("""
        <style>
        .big-font2{font-size:30px !important;color:yellow;text:align:center;}</style>
    """,unsafe_allow_html=True)
        if st.checkbox("Owner of the app"):
            st.markdown("""
        <style>
        .behre{font-size:20px;text-align:justify;font:weight:bold;}
        .behre1{font-size:200%;font-weight:bold;color:green;text-align:justify;
        }
        </style>
        """,unsafe_allow_html=True)
            image = Image.open('F:/New chapter/PG/DSA/Tutorials/Assignment/Multiple_disease/aa.jpg')
            st.image(image,width=300)
            st.title("Name : BEHREDIN REDI YILMA")
            st.markdown('<p class="behre1">Jimma Institute of Technology</p>',unsafe_allow_html=True)
            st.markdown('<p class="behre">Program : Msc in Artificial Intelligence 1st Year</p>',unsafe_allow_html=True)
            st.markdown('<p class="behre">Phone Number : 0932632526</p>',unsafe_allow_html=True)
            st.markdown('<p class="behre">Email : behreredi5497@gmail.com</p>',unsafe_allow_html=True)
            st.markdown('<p class="behre1">Academic Year : June 2023 GC</p>',unsafe_allow_html=True)