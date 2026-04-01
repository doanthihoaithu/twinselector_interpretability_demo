"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""
import os

import numpy as np
import plotly
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics._ranking import _ndcg_sample_scores

from st_files_connection import FilesConnection

# s3_conn = st.connection('s3', type=FilesConnection)
# df = s3_conn.read("interpretability-anomaly-scores-665163999694-us-east-2-an/sub_scores/settings_six/cblof/synthetic_batch_801.out", input_format="csv", ttl=600)
# print("S3 df shape", df.shape)

from utils.constant import list_measures, list_length, method_group, methods_ens, old_method, all_datasets
from utils.helper import generate_dataframe, plot_box_plot, add_rect, plot_batch_mts, \
	estimate_dimension_contribution_with_a_buffer, plot_interpretability_curves

# from models.run_model import run_model

# list_batches = ['batch_0.csv', 'batch_1.csv', 'batch_2.csv', 'batch_3.csv', 'batch_4.csv']

# @hydra.main(config_path="conf", config_name="config.yaml")
# def main(config: DictConfig):
	# mts_data_folder = config.mts_data_folder
	# mts_running_dataset = config.mts_running_dataset
	# print(mts_data_folder, mts_running_dataset)
# Page Configuration and Title
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
st.set_page_config(layout="wide")
st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 300px;
            margin-left: -300px;
        }
        """,
        unsafe_allow_html=True,
    )
st.markdown('# Interpretability Evaluation')
# st.markdown('Overall evaluation of 125 classification algorithms used for model selection for anomaly detection.
# We utilize 496 randomly selected time series from the TSB-UAD benchmark.')

mts_root_data_dir = 'data/mts'
datasets = os.listdir(mts_root_data_dir)
mts_infor_dict = {f: {
			'mts_data_dir' : f'data/mts/{f}/data',
			'mts_scores_dir' : f'data/mts/{f}/zip_sub_scores',
			'mts_merged_scores_dir' : f'data/mts/{f}/merged_scores/{f}'
		} for f in datasets}
# mts_data_dir = 'data/mts/settings_one/data'
# mts_scores_dir = 'data/mts/settings_one/scores'
# mts_merged_scores_dir = 'data/mts/settings_one/merged_scores/settings_one'

default_dataset = 'settings_six'
test_df = pd.read_csv(f'data/mts/{default_dataset}/merged_scores/{default_dataset}/current_inference_time.csv')
test_df.sort_values(by=['filename'], ascending=[True], inplace=True)
# testing_batch = [f'{f}.zip' for f in test_df['filename'].unique()]
testing_batch = ['synthetic_batch_0.out',
				'synthetic_batch_1.out',
				'synthetic_batch_10.out',
				'synthetic_batch_1001.out',
				'synthetic_batch_1002.out',
				'synthetic_batch_1003.out',
				'synthetic_batch_1004.out',
				'synthetic_batch_1005.out',
				'synthetic_batch_1007.out',
				'synthetic_batch_1008.out',
				'synthetic_batch_101.out',
				'synthetic_batch_1010.out',
				'synthetic_batch_1011.out',
				'synthetic_batch_1012.out']
testing_batch = [f'{f}.zip' for f in testing_batch]
mts_data_dir = mts_infor_dict[default_dataset]['mts_data_dir']
mts_scores_dir = mts_infor_dict[default_dataset]['mts_scores_dir']
list_batches = sorted([f for f in os.listdir(mts_data_dir) if 'multivariate_labels' not in f and f in testing_batch])
list_batches_multivariate_labels = [f for f in os.listdir(mts_data_dir) if 'multivariate_labels' in f and f in testing_batch]


# Loading data from CSV files
combined_metrics_of_base_detectors_df = pd.read_csv(f'data/mts/{default_dataset}/merged_scores/{default_dataset}/combined_results.csv')
list_algorithms = combined_metrics_of_base_detectors_df['algorithm'].unique().tolist()

new_interpretability_metrics_file = 'data/mts/settings_six/merged_scores/settings_six/combined_vus_pr_list.csv'
combined_interpretability_metrics_of_base_detectors_df = pd.read_csv(new_interpretability_metrics_file, index_col=0)
vus_pr_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if col.startswith('vus_pr')]
L_value_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if col not in vus_pr_columns and col.startswith('L_')]
combined_interpretability_metrics_of_base_detectors_df['INTERPRETABILITY_SCORE'] = combined_interpretability_metrics_of_base_detectors_df[vus_pr_columns].mean(axis=1)
combined_interpretability_metrics_of_base_detectors_df['FFVUS_PR'] = combined_interpretability_metrics_of_base_detectors_df.merge(combined_metrics_of_base_detectors_df[['algorithm','test_batch_id', 'FFVUS_PR']], on=['algorithm', 'test_batch_id'], how='left')['FFVUS_PR']

# Create tabs for displaying results
tab_overall, tab_explore = st.tabs(["Overall results", "Explore the results"])

with tab_overall:
	col_metric_over, col_dataset_over, col_method_over, col_length_over = st.columns([3, 1, 1, 1])

	# Metric selection
	with col_metric_over:
		metric_name = st.selectbox('Pick a measure',
								   list_measures,
								   help="Select the accuracy metric to evaluate the models.")

	# Dataset selection
	with col_dataset_over:
		datasets = st.multiselect('Pick datasets',
								  ['settings_six'],
								  default='settings_six',
								  help="Select one or more datasets for analysis.")

	# Method selection
	with col_method_over:
		methods_family = st.multiselect('Pick methods:',
										list(method_group.keys()),
										disabled=True,
										help="Select one or more method groups for comparison.")

	# Window length selection
	with col_length_over:
		length = st.multiselect('Pick window lengths:',
								list_length,
								disabled=True,
								help="Select the time window lengths applicable to the selected methods.")


	current_metric_df = combined_metrics_of_base_detectors_df[['algorithm', 'test_batch_id', metric_name]]

	oracle_metric = current_metric_df[
		current_metric_df['algorithm'].isin(old_method)].groupby(['test_batch_id'])[
		[metric_name]].max()
	oracle_metric['algorithm'] = 'oracle'
	oracle_metric.reset_index(inplace=True)

	current_metric_df = pd.concat([current_metric_df, oracle_metric], axis=0, ignore_index=True)

	df = pd.DataFrame()
	df['test_batch_id'] = combined_metrics_of_base_detectors_df['test_batch_id'].unique()
	for alg in current_metric_df['algorithm'].unique():
		metric_score_df = current_metric_df[current_metric_df['algorithm'] == alg][['test_batch_id', metric_name]]
		df[alg] = df.merge(metric_score_df, on='test_batch_id', how='left')[metric_name]

	df = df.set_index('test_batch_id')
	df.index.name = 'filename'
	df.sort_index(inplace=True)
	df['dataset'] = 'settings_six'

	# df = pd.read_csv(f'data/mts/{default_dataset}/merged_scores/{default_dataset}/current_accuracy_{metric_name}.csv')
	# df = df.set_index('filename')

	# Generate dataframe for plotting
	df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_score')
	st.dataframe(df_toplot, use_container_width=True)

	# Plot box plot using Plotly
	plot_box_plot(df_toplot, measure_name=metric_name)


# Tab for overall results with inline selection
with (tab_explore):
	# Setup columns for selecting metric, dataset, method, and window length
	col_dataset_exp, col_ts_exp, col_meth_exp, col_length_exp = st.columns([1, 1, 1, 1])

	# Metric selection
	with col_dataset_exp:
		dataset_exp = st.selectbox('Pick a dataset',
								   ['settings_six'],
								   help="Select a synthetic dataset to explore.")

	# Dataset selection
	with col_ts_exp:
		# datasets = st.multiselect('Pick datasets',
		# 						  all_datasets,
		# 						  help="Select one or more datasets for analysis.")
		batch_id = st.selectbox('Pick synthetic batch:',
								  list_batches,
								  help="Select one or more datasets for analysis.")
		batch_multivariate_labels = batch_id.replace('.zip', '.multivariate_labels.zip')

	# Window length selection
	with col_length_exp:
		length_selected_exp = st.selectbox('Pick a window length', list_length, disabled=True)

	# Method selection
	with col_meth_exp:
		method_selected_exp = st.selectbox('Pick a method', [meth.format(length_selected_exp) for meth in methods_ens],
										   disabled=True)



	path_ts = f'data/mts/{dataset_exp}/data/{batch_id}'
	path_ts_score = {AD_method: f'data/mts/{dataset_exp}/scores/{AD_method}/{batch_id}' for
					 AD_method in old_method}

	# Display the detector selected by the chosen method
	# st.markdown(
	# 	f"Detector selected by {method_selected_exp}: {df.at[batch_id.replace('.zip',''), method_selected_exp.replace('_score', '_class')]}")

	# Loading data from CSV files
	# df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
	# df = df.set_index('filename')

	batch_df = pd.read_csv(os.path.join(mts_data_dir, batch_id), header=0, compression='zip')
	batch_multivariate_labels_df = pd.read_csv(os.path.join(mts_data_dir, batch_multivariate_labels), header=0,  compression='zip')
	num_total_columns = batch_df.shape[1]
	sensor_columns = [f'Sensor{i}' for i in range(num_total_columns-1)]
	batch_df.columns = sensor_columns + ['is_anomaly']
	batch_multivariate_labels_df.columns = sensor_columns
	print(batch_df[sensor_columns].head())

	scores_dfs_dict = dict()
	contribution_dfs_dict = dict()
	ranking_scores_dfs_dict = dict()
	for alg in list_algorithms:
		# anomaly_score_path = os.path.join(mts_scores_dir, alg, batch_id)
		# # distribution_file_name = batch_id.replace('.zip', '.score_distribution.zip')
		# distribution_file_name = batch_id.replace('.out.zip', '.out.dimension_contribution.zip')
		# print('distributionfilename', distribution_file_name)
		# contribution_score_path = os.path.join(mts_scores_dir, alg, distribution_file_name)

		anomaly_score_path = os.path.join(mts_scores_dir, alg, batch_id)
		# distribution_file_name = batch_id.replace('.zip', '.score_distribution.zip')
		distribution_file_name = batch_id.replace('.out.zip', '.out.dimension_contribution.zip')
		print('distributionfilename', distribution_file_name)
		contribution_score_path = os.path.join(mts_scores_dir, alg, distribution_file_name)

		# df = s3_conn.read(
		# 	"interpretability-anomaly-scores-665163999694-us-east-2-an/sub_scores/settings_six/cblof/synthetic_batch_801.out",
		# 	input_format="csv", ttl=600)
		# print("S3 df shape", df.shape)

		# anomaly_score_path = f'interpretability-anomaly-scores-665163999694-us-east-2-an/sub_scores/settings_six/{alg}/{batch_id.replace(".zip", "")}'
		# # distribution_file_name = batch_id.replace('.zip', '.score_distribution.zip')
		# distribution_file_name = batch_id.replace('.out.zip', '.out.dimension_contribution')
		# # print('distributionfilename', distribution_file_name)
		# contribution_score_path = f'interpretability-anomaly-scores-665163999694-us-east-2-an/sub_scores/settings_six/{alg}/{distribution_file_name}'
		if os.path.exists(anomaly_score_path):
		# if True:
		# 	scores_dfs_dict[alg] = s3_conn.read(anomaly_score_path, input_format="csv", ttl=600)
		# 	raw_contribution_df = s3_conn.read(contribution_score_path, input_format="csv", ttl=600)

			# scores_dfs_dict[alg] = pd.read_csv(anomaly_score_path, header=None, sep='\s+')
			scores_dfs_dict[alg] = pd.read_csv(anomaly_score_path, header=None, sep=',')
			raw_contribution_df = pd.read_csv(contribution_score_path, sep=',')
			# contribution_dfs_dict[alg] = pd.read_csv(contribution_score_path, sep=',')

			estimated_dimension_contribution = estimate_dimension_contribution_with_a_buffer(
				raw_contribution_df.values,
				buffer=10)
			estimated_dimension_contribution = estimated_dimension_contribution / estimated_dimension_contribution.sum(
				axis=1, keepdims=True)
			contribution_dfs_dict[alg] = pd.DataFrame(estimated_dimension_contribution)
			# ranking_scores_dfs_dict[alg] = _ndcg_sample_scores(batch_multivariate_labels_df.values,
			# 												   estimated_dimension_contribution, k=5)
			ranking_scores_dfs_dict[alg] = np.zeros(batch_multivariate_labels_df.shape[0])
			# print(contribution_score_path)
			print("distribution.shape", contribution_dfs_dict[alg].shape)
			print("ranking_scores.shape", ranking_scores_dfs_dict[alg].shape)
		else:
			print(f"Path does not exist: {anomaly_score_path}")
	print(scores_dfs_dict.keys())
	# Generate dataframe for plotting
	# df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_score')
	# st.dataframe(df_toplot)


	colors = plotly.colors.qualitative.Plotly
	detector_color_map = {method: colors[i % len(colors)] for i, method in enumerate(old_method)}
	detector_color_map['oracle'] = 'black'
	detector_color_map['avg_ens'] = 'orange'
	detector_color_map['decision_tree_128_average_4'] = 'green'
	detector_color_map['decision_tree_256_preds'] = 'blue'

	# Plot box plot using Plotly
	plot_batch_mts(batch_id, batch_df[sensor_columns], batch_multivariate_labels_df,
				   scores_dfs_dict,
				   contribution_dfs_dict,
				   ranking_scores_dfs_dict,
				   detector_color_map)

	if batch_id.endswith('.zip'):
		batch_id = batch_id[:-4]
	# plot_interpretability_curves(batch_id, combined_interpretability_metrics_of_base_detectors_df, detector_color_map )

# Tab for exploring individual results
# with tab_explore:
# 	# Setup columns for selecting dataset, time series, method, and window length
# 	col_dataset_exp, col_ts_exp, col_meth_exp, col_length_exp = st.columns([1, 1, 1, 1])
#
# 	 # Dataset selection, including an option to upload custom dataset
# 	with col_dataset_exp:
# 		dataset_exp = st.selectbox('Pick a dataset', all_datasets + ['Upload your own'])
#
# 	# Time series selection based on the chosen dataset
# 	# with col_ts_exp:
# 	# 	time_series_selected_exp = st.selectbox('Pick a time series', list(df.loc[df['dataset'] == dataset_exp].index))
#
# 	# Window length selection
# 	with col_length_exp:
# 		length_selected_exp = st.selectbox('Pick a window length', list_length)
#
# 	# Method selection, showing methods tailored to selected window length
# 	with col_meth_exp:
# 		method_selected_exp = st.selectbox('Pick a method', [meth.format(length_selected_exp) for meth in methods_ens])

	# Custom dataset upload handling
	# if dataset_exp == 'Upload your own':
	# 	uploaded_ts = st.file_uploader("Upload your time series")
	# 	if uploaded_ts:
	# 		try:
	# 			# Process uploaded time series data
	# 			ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
	# 			ts_data = ts_data_raw[:, 0].astype(float)
	# 			ts_data = ts_data[:min(len(ts_data), 40000)]  # Limit data points to improve performance
	#
	# 			# Run model on the uploaded data
	# 			pred_detector, counter_dict = run_model(ts_data)
	# 			st.markdown("Voting results:")
	# 			st.bar_chart(counter_dict)
	# 			st.markdown(f"The Detector to select is {pred_detector}")
	#
	# 			# Plot time series and detected anomalies
	# 			trace_scores_upload = [go.Scattergl(x=list(range(len(ts_data))), y=ts_data,
	# 												mode='lines', line=dict(color='blue', width=3),
	# 												name="Time series", yaxis='y2')]
	# 			if len(ts_data_raw[0]) > 1:
	# 				label_data = ts_data_raw[:, 1]
	# 				label_data = label_data[:min(len(label_data), 40000)]
	# 				anom = add_rect(label_data, ts_data)
	# 				trace_scores_upload.append(go.Scattergl(x=list(range(len(ts_data))), y=anom,
	# 														mode='lines', line=dict(color='red', width=3),
	# 														name="Anomalies", yaxis='y2'))
	#
	# 			# Define layout for uploaded data plot
	# 			layout_upload = go.Layout(
	# 				yaxis=dict(domain=[0, 0.4], range=[0, 1]),
	# 				yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
	# 				title="Uploaded time series snippet (40k points maximum)",
	# 				template="simple_white",
	# 				margin=dict(l=8, r=4, t=50, b=10),
	# 				height=375,
	# 				hovermode="x unified",
	# 				xaxis=dict(range=[0, len(ts_data)])
	# 			)
	#
	# 			# Create and display the plot
	# 			fig = dict(data=trace_scores_upload, layout=layout_upload)
	# 			st.plotly_chart(fig, use_container_width=True)
	# 		except Exception as e:
	# 			st.error(f'File format not supported yet, please upload a time series in the TSB-UAD format: {e}')
	# else:
	# 	# Load data for the selected dataset and time series
	# 	path_ts = f'data/benchmark_ts/{dataset_exp}/{time_series_selected_exp}.zip'
	# 	path_ts_score = {AD_method: f'data/scores_ts/{dataset_exp}/{AD_method}/score/{time_series_selected_exp}.zip' for AD_method in old_method}
	#
	# 	# Display the detector selected by the chosen method
	# 	st.markdown(f"Detector selected by {method_selected_exp}: {df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')]}")
	#
	# 	# Load time series and anomaly data
	# 	ts_data_raw = pd.read_csv(path_ts, compression='zip', header=None).dropna().to_numpy()
	# 	label_data = ts_data_raw[:, 1]
	# 	ts_data = ts_data_raw[:, 0].astype(float)
	#
	# 	# Load scores for different methods and plot them
	# 	score_AD_method = pd.DataFrame()
	# 	for meth in path_ts_score.keys():
	# 		score_AD_method[meth] = pd.read_csv(path_ts_score[meth], compression='zip', header=None).dropna().to_numpy()[:, 0].astype(float)
	#
	# 	# Prepare and display plots for the data and scores
	# 	trace_scores = [go.Scattergl(x=list(range(len(ts_data))), y=ts_data, mode='lines', line=dict(color='blue', width=3), name="Time series", yaxis='y2'),
	# 					go.Scattergl(x=list(range(len(ts_data))), y=add_rect(label_data, ts_data), mode='lines', line=dict(color='red', width=3), name="Anomalies", yaxis='y2')]
	# 	for method_name in score_AD_method.columns:
	# 		alpha_val = 1 if method_name == df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')] else 0.05
	# 		trace_scores.append(go.Scattergl(x=list(range(len(ts_data))), y=[0] + list(score_AD_method[method_name].values[1:-1]) + [0],
	# 										 mode='lines', fill="tozeroy", name=f"{method_name} score", opacity=alpha_val))
	#
	# 	# Define layout for the plot
	# 	layout = go.Layout(
	# 		yaxis=dict(domain=[0, 0.4], range=[0, 1]),
	# 		yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
	# 		title=f"{time_series_selected_exp} time series snippet (40k points maximum)",
	# 		template="simple_white",
	# 		margin=dict(l=8, r=4, t=50, b=10),
	# 		height=375,
	# 		hovermode="x unified",
	# 		xaxis=dict(range=[0, len(ts_data)])
	# 	)
	#
	# 	# Create and display the plot
	# 	fig = dict(data=trace_scores, layout=layout)
	# 	st.plotly_chart(fig, use_container_width=True)