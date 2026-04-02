"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""

import streamlit as st
# from st_pages import Page, show_pages
from streamlit import Page

# from streamlit import Page

# from st_pages import Page, show_pages, add_page_title


# Specify what pages should be shown in the sidebar
# show_pages(
#     [
#         Page("app.py", "Overall Results", "🏠"),  # Home emoji is correct
#         # Page("pages/Accuracy.py", "Accuracy", "🎯"),  # Changed from :books: to a book emoji
#         Page("pages/Interpretability.py", "Interpretability", "🏠"),  # Changed to a brain emoji, more suitable for interpretability
#         # Page("pages/Datasets.py", "Datasets", "📊"),  # Changed to a chart emoji, more suitable for datasets
#         # Page("pages/Execution_Time.py", "Execution Time", "⏱️"),  # Changed to a stopwatch emoji, suitable for time
#         # Page("pages/Methods.py", "Methods", "🔧"),  # Changed to a tool emoji, suitable for methods or settings
#     ]
# )

# st.set_page_config(layout="wide")
pg = st.navigation([Page("pages/Overview.py"), Page("pages/Interpretability.py"),  Page("pages/Score.py")])
pg.run()



# Setup
# add_page_title() # Optional -- adds the title and icon to the current page
#
# st.title("Interpretability")
# # Show description of the Demo and main image
# st.markdown(description_intro)

# image_path = "figures/AUC_PR.png"
# image = Image.open(image_path)
# st.image(image, caption='AUC_PR of detectors on synthetic datasets')
#
# image_path = "figures/VUS_PR.png"
# image = Image.open(image_path)
# st.image(image, caption='VUS_PR of detectors on synthetic datasets')
#
# image_path = "figures/INTERPRETABILITY_HIT_2_SCORE.png"
# image = Image.open(image_path)
# st.image(image, caption='INTERPRETABILITY_HIT_2_SCORE of detectors on synthetic datasets')
#
#
# image_path = "figures/INTERPRETABILITY_LOG_SCORE.png"
# image = Image.open(image_path)
# st.image(image, caption='INTERPRETABILITY_LOG_SCORE of detectors on synthetic datasets')


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