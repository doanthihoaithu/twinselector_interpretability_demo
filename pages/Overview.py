import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")
    # css = '''
    # <style>
    #     [data-testid="stSidebar"]{
    #         min-width: 400px;
    #         max-width: 800px;
    #     }
    # </style>
    # '''
    # st.markdown(css, unsafe_allow_html=True)
st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 300;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 300px;
            margin-left: -300px;
        }
        """,
        unsafe_allow_html=True,
    )

st.markdown('# Overview')
st.markdown('The following figures provide a visual summary of the average VUS-PR and AUC-PR, along with interpretability metrics (Hit2 and KL divergence), across different components.')
st.markdown('These components include: (i) eight individual multivariate detectors; (ii) the Oracle (shown in white), representing the theoretical upper bound of model selection performance; (iii) Averaging Ensembling (shown in orange), which combines detector outputs via simple averaging; and (iv) Best Time Series Classification (shown in blue), corresponding to the best-performing multivariate time series classification model used for model selection.')
st.markdown('Click tab Interpretability for interactive visualization')



image_path = "figures/AUC_PR.png"
image = Image.open(image_path)
st.image(image, caption='AUC_PR of detectors on synthetic datasets', use_column_width=True)

image_path = "figures/VUS_PR.png"
image = Image.open(image_path)
st.image(image, caption='VUS_PR of detectors on synthetic datasets', use_column_width=True)

image_path = "figures/INTERPRETABILITY_HIT_2_SCORE.png"
image = Image.open(image_path)
st.image(image, caption='INTERPRETABILITY_HIT_2_SCORE of detectors on synthetic datasets', use_column_width=True)


# image_path = "figures/INTERPRETABILITY_LOG_SCORE.png"
# image = Image.open(image_path)
# st.image(image, caption='INTERPRETABILITY_LOG_SCORE of detectors on synthetic datasets')