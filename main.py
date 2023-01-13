import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from fer import FER
from keras.models import model_from_json



def AnalyseImage(img_array):

    test_image_one = img_array
    emo_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    #captured_emotions = emo_detector.detect_emotions(test_image_one)
    #dominant_emotion=max(captured_emotions.values())
    #st.write(dominant_emotion)
    # Print all captured emotions with the image
    emotion, score = emo_detector.top_emotion(test_image_one)
    print(emotion,score)
    #st.write(captured_emotions)
    plt.imshow(test_image_one)
    return emotion,score

def main():
    # Face Analysis Application #
    st.title("Heerop Kafae")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Emir Jibran Badardin
             Email : mirjibs@gmail.com""")

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        #food, service and atomosphere overall=
        st.write("""
                 The application has two functionalities.

                 1. Detect Customer Emotion .

                 2. Customer Reviews.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam To Capture Image")
        st.write("Click on start to use webcam and detect your face emotion")
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img_array = np.array(img)

            # Check the type of img_array:
            # Should output: <class 'numpy.ndarray'>
            st.write(type(img_array))

            # Check the shape of img_array:
            # Should output shape: (height, width, channels)
            st.write(img_array.shape)


            st.write(AnalyseImage(img_array))






    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass

if __name__ == '__main__':
    main()


