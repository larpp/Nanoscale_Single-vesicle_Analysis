import cv2
import streamlit as st
import numpy as np
import pandas as pd
import os
import zipfile
import io
import shutil
import ray
from util import draw_bbox_array, xywh2xyxy
from infer import online_infer, batch_infer


def main():
    st.title("Clustering")
    if "ray" not in st.session_state:
        st.session_state.ray = False

    if st.session_state.ray == True:
        st.session_state.ray = False
        ray.shutdown()

    with st.sidebar:
        sic = st.checkbox("Show Inference confidence")
        conf_thres = st.slider("conf_thres", 0.0, 1.0, 0.4, 0.01)
        iou_thres = st.slider("iou_thres", 0.0, 1.0, 0.45, 0.01)

    tab1, tab2, tab3 = st.tabs(["Inference", "Batch Inference", "Train model result"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), flags=1)

            col1, col2 = st.columns(2)
            with col1:
                st.write("원본")
                st.image(image)

            with col2:
                draw_img_array, csv = online_infer(image, conf_thres, iou_thres, img_shape=(640, 640), sic=sic)
                st.write("결과")
                st.image(draw_img_array)
                st.download_button(
                    label="Download result CSV",
                    data=pd.DataFrame(csv).to_csv(index=False).encode("utf-8"),
                    file_name=f"{'.'.join(uploaded_file.name.split('.')[:-1])}_result.csv",
                )

    with tab2:
        with st.form("my-form", clear_on_submit=True):
            uploaded_zip = st.file_uploader("Choose an image ZIP", type=["zip"])
            submitted = st.form_submit_button("파일 분석")
            if submitted:
                if os.path.exists("inputdata"):
                    shutil.rmtree("inputdata", ignore_errors=True)
                if uploaded_zip is not None:
                    with zipfile.ZipFile(uploaded_zip, "r") as z:
                        z.extractall(f"inputdata/{'.'.join(z.filename.split('.')[:-1])}")

        if os.path.exists("inputdata"):
            my_bar = st.progress(0.0, "분석중입니다.")
            img_path_list = [
                os.path.join(os.getcwd(), root, file_name)
                for root, _, files in os.walk("inputdata")
                for file_name in files
                if os.path.splitext(file_name)[-1] in [".jpg", ".jpeg", ".png"]
            ]
            batch_size, n = 2, len(img_path_list)

            if st.session_state.ray == False:
                st.session_state.ray = True
                ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "./streamlit_frontend"})
            batch_result = [
                batch_infer.remote(
                    img_path_list[idx : idx + batch_size],
                    conf_thres,
                    iou_thres,
                    (640, 640),
                    os.path.join(os.getcwd(), "inputdata"),
                )
                for idx in range(0, n, batch_size)
            ]
            buf, N = io.BytesIO(), len(batch_result)
            with zipfile.ZipFile(buf, "x") as csv_zip:
                while len(batch_result):
                    done, batch_result = ray.wait(batch_result)
                    mini_batch_result = ray.get(done[0])
                    my_bar_per = 1 - len(batch_result) / N
                    my_bar.progress(my_bar_per, text="분석중입니다. : " + str(int(my_bar_per * 100)).zfill(2) + "% / 100%")

                    for csv, csv_name in zip(*mini_batch_result):
                        csv_zip.writestr(csv_name, pd.DataFrame(csv).to_csv(index=False))

            if st.session_state.ray == True:
                st.session_state.ray = False
                ray.shutdown()

            st.download_button(
                label="Download result zip",
                data=buf.getvalue(),
                file_name=f"{os.listdir('inputdata')[0]}_conf({conf_thres:.2f})_iou({iou_thres:.2f}).zip",
                mime="application/zip",
            )
            shutil.rmtree("inputdata", ignore_errors=True)

    with tab3:
        data_type = st.radio(
            "Set selectbox data_type",
            options=["train", "valid", "test"],
        )
        image_path = f"./data/{data_type}/images"
        label_path = f"./data/{data_type}/labels"

        img_path = st.selectbox(
            "image를 선택해주세요.",
            os.listdir(image_path),
        )
        agree = st.checkbox("추론 결과를 원한다면 체크해주세요.")
        tab2_col1, tab2_col2, tab2_col3 = st.columns(3)
        with tab2_col1:
            st.write("원본")
            img = cv2.imread(os.path.join(image_path, img_path))
            st.image(img)

        with tab2_col2:
            st.write("원본 bbox")
            label = img_path.replace("jpg", "txt")
            df = pd.read_table(
                os.path.join(label_path, label),
                sep=" ",
                header=None,
                index_col=0,
            )
            det = np.append(xywh2xyxy(df.values), [[1, 0]] * len(df.values), axis=1)
            draw_img_array, det = draw_bbox_array(det, (1, 1), img, sic)
            st.image(draw_img_array)

        with tab2_col3:
            st.write("추론 bbox")
            if agree:
                draw_img_array, _ = online_infer(img, conf_thres, iou_thres, img_shape=(640, 640), sic=False)
                st.image(draw_img_array)


if __name__ == "__main__":
    main()
