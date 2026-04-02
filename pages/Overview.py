import streamlit as st
from PIL import Image

# st.set_page_config(layout="wide")
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
st.markdown('The following figures provide a visual summary of the average VUS-PR and AUC-PR, along with interpretability metrics (Hit@1), across different components.')
st.markdown('These components include: \n'
            '* (i) ten individual multivariate detectors; \n'
            '* (ii) the Oracle (shown in white), representing the theoretical upper bound of model selection performance; \n'
            '* (iii) Averaging Ensembling (shown in orange), which combines all base detectors via simple averaging; \n'
            '* (iv) Model Selector (shown in blue) which select only one detector for a given MTS; and \n'
            '* (v) Model Selector with detector combination (shown in green).'
            )
st.markdown('Click tab Interpretability for interactive visualization, select tab **Explore the results** for visualizing MTS and their corresponding anomaly scores. Note that this tab shows only some testing samples.')



image_path = "figures/AUC_PR.png"
image = Image.open(image_path)
st.image(image, caption='AUC_PR of detectors on synthetic datasets', use_column_width=True)

image_path = "figures/VUS_PR.png"
image = Image.open(image_path)
st.image(image, caption='VUS_PR of detectors on synthetic datasets', use_column_width=True)

image_path = "figures/INTERPRETABILITY_CONDITIONAL_HIT_1_SCORE.png"
image = Image.open(image_path)
st.image(image, caption='INTERPRETABILITY_CONDITIONAL_HIT_1_SCORE of detectors on synthetic datasets', use_column_width=True)

image_path = "figures/DUAL_MODEL_SELECTOR.png"
image = Image.open(image_path)
st.image(image, caption='TWO MODEL SELECTORS FOR ACCURACY AND INTERPRETABILITY', use_column_width=True)


# image_path = "figures/INTERPRETABILITY_LOG_SCORE.png"
# image = Image.open(image_path)
# st.image(image, caption='INTERPRETABILITY_LOG_SCORE of detectors on synthetic datasets')