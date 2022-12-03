from os import listdir
import av
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
class_name = ['00000','1000 VND','10.000 VND','100.000 VND','2000 VND','20.000 VND','5000 VND','50.000 VND','500.000 VND']
# load model
def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(9, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights
my_model = get_model()
my_model.load_weights("weights-48-0.97.hdf5")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        origin_img = frame.to_ndarray(format="bgr24")
        image = origin_img.copy()
        image = cv2.resize(image, dsize=(128, 128))
        image = image.astype('float')*1./255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)       

        # Predict
        predict = my_model.predict(image)
        print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
        print(np.max(predict[0],axis=0))
        if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):


            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv2.putText(origin_img, class_name[np.argmax(predict)], org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Picture", origin_img)
        return av.VideoFrame.from_ndarray(origin_img, format="bgr24")

def main():
    # Face Analysis Application #
    st.title("Money detect app")
    activities = ["About", "Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    if choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                        Chương trình sử dụng open-cv để nhận diện tiền Việt Nam
                                    </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">
                                        Nguyễn Huỳnh Minh Trung MSSV: 19133061
                                        Nguyễn Thế Ngọc MSSV: 19133040
                                    </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
    else:
        pass


if __name__ == "__main__":
    main()
