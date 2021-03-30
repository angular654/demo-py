# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from PIL import Image
import os
from imageai.Detection import ObjectDetection


def save_uploadedfile(uploadedfile):
    with open(os.path.join("Data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


def recognition(image):
        exec_path = os.getcwd()
        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(
        exec_path, "yolo.h5")
        )
        detector.loadModel()
        detections = detector.detectObjectsFromImage(
            input_image=os.path.join("Data", image),
            output_image_path=os.path.join(exec_path, "out.jpg"),
            minimum_percentage_probability=70,
            display_percentage_probability=True,
            display_object_name=True
        )
        for DetectedObject in detections:
                st.write(DetectedObject["name"], " : ", DetectedObject["percentage_probability"])
        return True


def main():
    st.sidebar.header("Snake vision")
    st.sidebar.image("sidebar-logo.jpg")
    st.title("Распознавание объектов")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        imageLocation = st.empty()
        imageLocation.image(image, caption=uploaded_file.name, use_column_width=True)
        st.write("")
        save_uploadedfile(uploaded_file)
        btn = st.empty()
        btn.button('Начать распознование')
        if(btn):
            if(recognition(uploaded_file.name)):
                imageLocation.image("out.jpg")
                os.remove(f'./Data/{uploaded_file.name}')
                btn.success("Готово!")
                
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
