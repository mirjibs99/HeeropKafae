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
        # Create dataframe to store the review data
        data = {'Food': [0], 'Service': [0], 'Atmosphere': [0]}
        df = pd.DataFrame(data)

        # Capture image for food category
        st.write("For food category:")
        st.write("Please show your emotion for the Food category")
        image_bytes_food = st.camera_input("Food Review")
        if image_bytes_food is not None:
            # To read image file buffer as a PIL Image:
                img = Image.open(image_bytes_food)
                food_array = np.array(img)
                st.write(type(food_array))
                st.write(food_array.shape)
                st.write(AnalyseImage(food_array))
                emotion_food=AnalyseImage(food_array)
                emotionfood=str(emotion_food[0])
                print(emotionfood)
                # if st.button('Submit', key='food_button'):
                #     st.write("Thank you for your review!")
                if emotion_food[0] == 'happy':
                    df['Food'][0] += 4
                elif emotion_food[0] == 'neutral':
                    df['Food'][0] += 3
                elif emotion_food[0] == 'sad':
                    df['Food'][0] += 2
                elif emotion_food[0] == 'angry':
                    df['Food'][0] += 1

        # if st.button('Submit', key='food_button'):
        #         st.write("Thank you for your review!")
        #         st.write(emotion_food)
        #         df.to_csv('review_data.csv')

        #df.to_csv('review_data.csv')


            # Capture image for service category
        st.write("For service category:")
        image_bytes_service = st.camera_input("Service Review")
        if image_bytes_service is not None:
            # To read image file buffer as a PIL Image:
                img = Image.open(image_bytes_service)
                service_array = np.array(img)
                st.write(type(service_array))
                st.write(service_array.shape)
                emotion_service= AnalyseImage(service_array)
                # if st.button('Submit', key='service_button'):
                #     st.write("Thank you for your review!")
                if emotion_service[0] == 'happy':
                    df['Service'][0] += 4
                if emotion_service[0] == 'neutral':
                    df['Service'][0] += 3
                if emotion_service[0] == 'sad':
                    df['Service'][0] += 2
                if emotion_service[0] == 'angry':
                    df['Service'][0] += 1
                st.write(emotion_service)

                # try:
                #     old_df = pd.read_csv('review_data.csv')
                #     df = pd.concat([old_df, df], ignore_index=True)
                # except FileNotFoundError:
                #     pass
                # df.to_csv('review_data.csv', index=False)

        # Capture image for atmosphere category
        st.write("For atmosphere category:")
        image_bytes_atmosphere = st.camera_input("Atmosphere Review")
        if image_bytes_atmosphere is not None:
            # To read image file buffer as a PIL Image:
                img = Image.open(image_bytes_atmosphere)
                atmosphere_array = np.array(img)
                st.write(type(atmosphere_array))
                st.write(atmosphere_array.shape)
                #st.write(AnalyseImage(atmosphere_array))
                emotion_atmosphere = AnalyseImage(atmosphere_array)
                # if st.button('Submit', key='atmosphere_button'):
                #     st.write("Thank you for your review!")
                if emotion_atmosphere[0] == 'happy':
                    df['Atmosphere'][0] += 4
                if emotion_atmosphere[0] == 'neutral':
                    df['Atmosphere'][0] += 3
                if emotion_atmosphere[0] == 'sad':
                    df['Atmosphere'][0] += 2
                if emotion_atmosphere[0] == 'angry':
                    df['Atmosphere'][0] += 1
                st.write(emotion_atmosphere)
        new_data = {'Food': str(df['Food'][0]), 'Service': str(df['Service'][0]), 'Atmosphere': str(df['Atmosphere'][0])}

        if st.button('Submit', key='food_button'):
            st.write("Thank you for your review!")
            st.write("Your predicted emotion for our food is "+emotionfood+" with a rating of "+str(df['Food'][0])+"⭐")
            st.write("Your predicted emotion for our service is " + str(emotion_service[0]) + " with a rating of " + str(df['Service'][0]) + "⭐")
            st.write("Your predicted emotion for our atmosphere is " + str(emotion_atmosphere[0]) + " with a rating of " + str(df['Atmosphere'][0]) + "⭐")
            df = pd.read_csv('concat.csv')
            df1=pd.concat([df,pd.DataFrame(new_data, index=[0])],axis=0,ignore_index=True,join='inner')
            df1.to_csv('concat.csv')
            # df = df.append(new_data, ignore_index=True)
            # df.to_csv('review_data.csv')




        # img_file_buffer = st.camera_input("Take a picture")
        #
        # if img_file_buffer is not None:
        #     # To read image file buffer as a PIL Image:
        #     img = Image.open(img_file_buffer)
        #
        #     # To convert PIL Image to numpy array:
        #     img_array = np.array(img)
        #
        #     # Check the type of img_array:
        #     # Should output: <class 'numpy.ndarray'>
        #     st.write(type(img_array))
        #
        #     # Check the shape of img_array:
        #     # Should output shape: (height, width, channels)
        #     st.write(img_array.shape)
        #
        #
        #     st.write(AnalyseImage(img_array))









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
                             		<h4 style="color:white;text-align:center;">This Application is developed by Emir Jibran using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.  </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass

if __name__ == '__main__':
    main()


