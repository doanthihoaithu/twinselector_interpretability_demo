"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""
import streamlit as st
from PIL import Image
import pandas as pd
# from st_pages import Page, show_pages
from streamlit import Page

# from streamlit import Page

from pages import Interpretability
# from st_pages import Page, show_pages, add_page_title

from utils.constant import description_intro, list_measures, list_length, method_group, template_names
from utils.helper import init_names


# @hydra.main(config_path="conf", config_name="config.yaml")
def main():
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
            width: 100px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 100px;
            margin-left: -100px;
        }
        """,
        unsafe_allow_html=True,
    )
# Specify what pages should be shown in the sidebar
#     show_pages(
#         [
#             Page("app.py", "Overall Results", "🏠"),  # Home emoji is correct
#             # Page("pages/Accuracy.py", "Accuracy", "🎯"),  # Changed from :books: to a book emoji
#             Page("pages/Interpretability.py", "Interpretability", "🏠"),  # Changed to a brain emoji, more suitable for interpretability
#             # Page("pages/Datasets.py", "Datasets", "📊"),  # Changed to a chart emoji, more suitable for datasets
#             # Page("pages/Execution_Time.py", "Execution Time", "⏱️"),  # Changed to a stopwatch emoji, suitable for time
#             # Page("pages/Methods.py", "Methods", "🔧"),  # Changed to a tool emoji, suitable for methods or settings
#         ]
#     )

    pg = st.navigation([Page("pages/Overview.py"), Page("pages/Score.py"), Page("pages/Interpretability_Test.py"), Page("pages/Interpretability_Full.py")])
    pg.run()



    # Setup
	# add_page_title() # Optional -- adds the title and icon to the current page
	#
	# st.title("Interpretability")
	# # Show description of the Demo and main image
	# st.markdown(description_intro)


	# page_names_to_funcs = {
	# 		# "—": intro,
	# 		# "Plotting Demo": plotting_demo,
	# 		# "Mapping Demo": mapping_demo,
	# 		# "DataFrame Demo": data_frame_demo
	# 		'Interpretability': Page("pages/Interpretability.py"),
	# 	}
	#
	# # demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
	# page_names_to_funcs['Interpretability']()



	# try:
	# 	image_path = 'figures/3_pipeline.jpg'
	# 	image = Image.open(image_path)
	# 	st.image(image, caption='Overview of the model selection pipeline')
	# except FileNotFoundError:
	# 	st.error(f"Error: The file {image_path} does not exist.")
	#
	#
	# # Loading data from CSV files
	# df = pd.read_csv('data/merged_scores_{}.csv'.format('VUS_PR'))
	# df = df.set_index('filename')
	#
	# df_time = pd.read_csv('data/inference_time.csv')
	# df_time = df_time.rename(columns={'Unnamed: 0': 'filename'})
	# df_time = df_time.set_index('filename')
	#
	# df_time_train = pd.read_csv('data/training_times.csv', index_col='window_size')
	# final_names = init_names(list_length, template_names)


if __name__ == '__main__':
    main()