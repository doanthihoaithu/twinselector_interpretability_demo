"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""

import kaleido #required
import streamlit

print('kaleido version:', kaleido.__version__) #0.2.1


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numpy.random import default_rng as rng
import plotly.figure_factory as ff
import numpy as np


import matplotlib.colors as mcolors

from .constant import methods_colors, methods_conv, methods_sit, methods_ts, methods_classical, old_method, list_length, \
	method_group, template_names, best_ms_combination, best_ms_selection, baselines


def init_names(list_length, template_names):
	"""
	Initializes and returns a dictionary with template names adjusted to the given list lengths.

	Args:
		list_length (list): List of lengths for which template names need adjustment.
		template_names (dict): Dictionary containing template names as keys.

	Returns:
		dict: A dictionary with adjusted template names based on the list lengths.

	Example:
		If list_length = [10, 20] and template_names = {'template_{}': 'value'}, 
		the function will return {'template_10': 'value10', 'template_20': 'value20'}.
	"""
	final_names = {}
	for length in list_length:
		for key, value in template_names.items():
			if '{}' in key:
				new_key = key.format(length)
				new_value = value.format(length)
				final_names[new_key] = new_value
			else:
				final_names[key] = value

	return final_names


def plot_box_plot(df, measure_name, scale='linear'):
	"""
	Plots a box plot using Seaborn and Matplotlib.

	Args:
		df (DataFrame): DataFrame containing data to be plotted.
		measure_name (str): Name of the measure for labeling the x-axis.
		scale (str, optional): Scale for the x-axis ('linear' or 'log'). Defaults to 'linear'.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create color palette
	my_pal = {method: methods_colors["detectors"] for method in old_method}
	my_pal.update({method: methods_colors['best_ms_selection'] for method in best_ms_selection})
	my_pal.update({method: methods_colors['best_ms_combine'] for method in best_ms_combination})
	my_pal.update({"Avg Ens": methods_colors["avg_ens"], 'Oracle': methods_colors["oracle"]})
	my_pal.update({"avg_ens": methods_colors["avg_ens"], 'oracle': methods_colors["oracle"]})

	for family, color in zip([methods_conv, methods_sit, methods_ts, methods_classical],
							 [methods_colors["conv"], methods_colors["sit"], methods_colors["rocket"],
							  methods_colors["feature_based"]]):
		for length in list_length:
			my_pal_tmp = {method.format(length): color for method in family}
			my_pal = {**my_pal, **my_pal_tmp}
	# Add same names with '_inf' instead of '_score'
	my_pal_plus = {}
	
	for key in my_pal.keys():
		my_pal_plus[key] = my_pal[key]
		my_pal_plus[key.replace('_score', '_inf')] = my_pal[key]
		my_pal_plus[key.replace('_score', '')] = my_pal[key]
		my_pal_plus[key.replace('_default', '')] = my_pal[key]
		my_pal_plus[key.replace('_default', '').replace('_score', '')] = my_pal[key]

	# # Melting dataframe to long format for better handling by Plotly Express
	# df_long = df.melt(var_name='Variable', value_name=measure_name)

	# # Determine order for box plot based on median values
	# order = {"Variable": list(df.median().sort_values().index)[::-1]}

	# # Create box plot using Plotly
	# fig = px.box(
	# 	df_long, 
	# 	x=measure_name, 
	# 	y='Variable',
	# 	category_orders=order,
	# 	color='Variable',
	# 	orientation='h',
	# 	title=f'Box Plot of {measure_name}',
	# 	color_discrete_map=my_pal_plus,
	# 	points='all',
	# )

	# # Adjust x-axis scale if specified
	# if scale == 'log':
	# 	fig.update_xaxes(type='log')

	# # Update layout and styles
	# fig.update_layout(
	# 	height=max(400, 30 * len(df.columns)),  # Dynamic height based on number of columns
	# 	width=800,  # Suitable width to handle large labels and readability
	# 	showlegend=False  # Optional: Hide legend if not necessary
	# )

	# # Customize hover data
	# fig.update_traces(hoverinfo='x+name')

	# # Display the plot in Streamlit
	# st.plotly_chart(fig, use_container_width=True)

	# Determine plot dimensions based on DataFrame length
	fig_height = min(30, max(1, int(0.40 * len(df.columns))))
	fig = plt.figure(figsize=(10, fig_height))

	# Determine order for box plot based on median values
	order = list(df.median().sort_values().index)[::-1]

	# Create box plot
	g = sns.boxplot(
		data=df, 
		order=order, 
		palette=my_pal_plus, 
		showfliers=False, 
		orient='v',
		saturation=1, 
		whis=0.241289844
	)
	
	g.xaxis.grid(True)
	plt.ylabel(measure_name)
	plt.xticks(rotation=90)
	if scale == 'log':
		plt.xscale('log')
	st.pyplot(fig)

def process_anomaly_index_to_windows(label_ts):
	windows = []
	# current_anomaly_window = (label_ts[0], label_ts[0] + window_size)
	current_start = None
	current_end = None

	for index, label in label_ts.items():
		# print(index, label)
		if label == 0:
			if current_end is None:
				current_start = None
				current_end = None
			elif current_end is not None:
				windows.append((current_start, current_end))
				current_start = None
				current_end = None
		elif label == 1:
			if current_end is None:
				current_start = index
				current_end = index
			else:
				assert current_start != None
				current_end = index
	# print("Anomaly windows", windows)
	return windows

def plot_batch_mts_simple(batch_id, df, multivariate_labels_df, scores_dfs_dict, contribution_dfs_dict,
				   ranking_scores_dfs_dict,
				   detector_color_map):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create subplots for each time series
	num_series = len(df.columns)

	# fig = make_subplots(rows=num_series + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

	data = []
	# Add traces for each time series
	# for i, col in enumerate(df.columns):
	# 	# print(i, col)
	# 	# print(df.shape)
	#
	# 	# fig.add_trace(
	# 	# 	go.Scatter(x=df.index.to_list(), y=df[col].to_list(), mode='lines', name=col),
	# 	# 	row=i + 1,
	# 	# 	col=1
	# 	# )
	#
	# 	data.append(go.Scatter(x=df.index.to_list(), y=df[col].to_list(),
	# 						   mode='lines',
	# 						   name=col,
	# 						   xaxis='x',
	# 						   yaxis='y' if i == 0 else f'y{i+1}',
	# 						   legendgroup='mts_data',
	# 						   # visible='legendonly',
	# 						   ),)
	# 	# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
	# 	# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
	# 	# 					   mode='markers',
	# 	# 					   fillcolor= 'red',
	# 	# 					   name=col,
	# 	# 					   xaxis='x',
	# 	# 					   yaxis='y' if i == 0 else f'y{i + 1}'
	# 	# 					   ), )

	is_anomaly_ts = multivariate_labels_df.sum(axis=1) > 0
	data.append(go.Scatter(x=df.index.astype(int).to_list(), y=is_anomaly_ts.astype(int).to_list(),
						   mode='lines',
						   fillcolor= 'black',
						   name='Anomaly Label',
						   xaxis='x',
						   yaxis='y',
						   legendgroup='univariate_anomaly_label',
						   ),
				)

	if len(scores_dfs_dict.keys()) > 0:
		# Add traces for each score DataFrame
		for method_name, scores_df in scores_dfs_dict.items():
			ranking_scores_numpy = ranking_scores_dfs_dict[method_name]
			color = detector_color_map.get(method_name, 'black')  # Default to black if method name not found in color map
			# print('Color for method', method_name, color)
			contribution_df = contribution_dfs_dict[method_name]
			customdata = []
			ranking_customdata = []
			# print('Contribution.shape', contribution_df.shape)
			# print('Score shape', scores_df.shape)
			# print(f'Contribution df shape', contribution_df.shape)

			# for row_index, row in contribution_df.iterrows():
			# 	# print(f'Row index {row_index}', row)
			# 	dimensional_label = multivariate_labels_df.iloc[row_index].values
			# 	top_1_id = int(row[2])
			# 	top_2_id = int(row[1])
			# 	hit_k_score = row[0]
			# 	# print(row_index, row)
			# 	# str_list = f'<b>Score:{hit_k_score:.2f}--s{top_1_id}:s{top_2_id}</b>'
			# 	str_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>:{f:.2f}' for i,f in enumerate(row.values.tolist())])
			# 	ranking_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>' for i in (-row).argsort().values.tolist()])
			# 	customdata.append(str_list)
			# 	ranking_customdata.append(ranking_list)

			# print('Contribution shape', len(customdata))
			# fig.add_trace(
			# 	go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
			# 			   mode='lines', name=f"{method_name} score"),
			# 	row=num_series + 1,
			# 	col=1
			# )
			data.append(go.Scatter(x=scores_df.index.astype(int).to_list(), y=scores_df[scores_df.columns[0]].to_list(),
						   mode='lines', name=f"{method_name} score", xaxis='x', yaxis=f'y2',
								   # customdata=['a:1, b:2, c:3'] * len(scores_df),
								   # customdata=customdata,
								   line=dict(color=color),
								   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
								   # hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}",
								   legendgroup=method_name
								   ),)

			data.append(go.Scatter(x=scores_df.index.astype(int).to_list(), y=ranking_scores_numpy.tolist(),
								   mode='lines', name=f"{method_name} NDCG@k", xaxis='x', yaxis=f'y3',
								   # customdata=['a:1, b:2, c:3'] * len(scores_df),
								   # customdata=ranking_customdata,
								   # hovertemplate="%{y:.4f}<br><b>Ranking</b>: %{customdata}",
								   # hovertemplate="%{y:.4f}",
								   line=dict(color=color),
								   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
								   # hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}"
								   legendgroup=method_name
								   ), )

	layout = dict(
		# height=100 * (num_series + 1),
		height=200 * 3,
		showlegend=True,
		hoversubplots="axis",
		title=dict(text=f'Anomaly scores and Interpretability scores (NDCG@k) of detectors are shown in the last two subplots, with their contribution to the score in the hover.'),
		hovermode="x unified",
		grid=dict(rows=3, columns=1),
		# yaxis=dict(title=df.columns[0]),
		# yaxis1=dict(title=df.columns[1])
		# yaxes=[dict(title=f, showgrid=True, zeroline=False, showline=True, ticks='outside', row=num_series+1, col=1) for f in df.columns],
		# use_container_width=True
	)


	fig = go.Figure(data=data, layout=layout, _validate=False)
	# for i, col in enumerate(df.columns):
	# 	# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
	# 	xref = 'x' if i == 0 else f'x{i+1}'
	# 	yref = 'y' if i == 0 else f'y{i+1}'
	# 	# for anomaly_index in anomaly_ts.index.to_list():
	# 	anomaly_windows = process_anomaly_index_to_windows(multivariate_labels_df[col])
	# 	for start, end in anomaly_windows:
	# 		# fig.add_vrect(x=anomaly_ts.index.to_list()[0], line_dash='solid', line_color='red', opacity=0.2, xref='x',
	# 		# 			  yref=yref)
	# 		fig.add_vrect(x0=start, x1=end, fillcolor='red', line_color='red', opacity=0.2, xref='x', yref=yref)
	# 	# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
	# 	# 					   mode='markers',
	# 	# 					   fillcolor='red',
	# 	# 					   name=col,
	# 	# 					   xaxis='x',
	# 	# 					   yaxis='y' if i == 0 else f'y{i + 1}'
	# 	# 					   ), )

	# for i, col in enumerate(df.columns):
	# 	fig.update_layout(**{f'yaxis{i+1}': dict(title=col, showgrid=True, zeroline=False, showline=True, ticks='outside')})

	fig.update_layout(**{f'yaxis1': dict(title='Label', showgrid=True, zeroline=False, showline=True, ticks='outside',
										 # tickangle=30
										 )})
	fig.update_layout(**{f'yaxis2': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
													   # tickangle=30
													   )})
	fig.update_layout(**{
		f'yaxis3': dict(title='NDCG@k', showgrid=True, zeroline=False, showline=True, ticks='outside',
										# tickangle=30
										)})

	# fig.write_html(f'html/{batch_id}_mts_vs_scores.html')  # Save the figure to a file
	# print(f"Saved interactive plot for batch {batch_id} at html/{batch_id}_mts_vs_scores.html")
	# fig.write_image(f'images/{batch_id}_mts_vs_scores.png', scale=1.0)  # Save the figure to a file
	# print(f"Saved static image for batch {batch_id} at images/{batch_id}_mts_vs_scores.png")

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True, key='plot_mts_simple', validate=False)
def plot_batch_mts(batch_id, df, multivariate_labels_df, scores_dfs_dict, contribution_dfs_dict,
				   ranking_scores_dfs_dict,
				   detector_color_map):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create subplots for each time series
	num_series = len(df.columns)

	# fig = make_subplots(rows=num_series + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

	data = []
	# Add traces for each time series
	for i, col in enumerate(df.columns):
		# print(i, col)
		# print(df.shape)

		# fig.add_trace(
		# 	go.Scatter(x=df.index.to_list(), y=df[col].to_list(), mode='lines', name=col),
		# 	row=i + 1,
		# 	col=1
		# )

		data.append(go.Scatter(x=df.index.to_list(), y=df[col].to_list(),
							   mode='lines',
							   name=col,
							   xaxis='x',
							   yaxis='y' if i == 0 else f'y{i+1}',
							   legendgroup='mts_data',
							   # visible='legendonly',
							   ),)
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor= 'red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	if len(scores_dfs_dict.keys()) > 0:
		# Add traces for each score DataFrame
		for method_name, scores_df in scores_dfs_dict.items():
			ranking_scores_numpy = ranking_scores_dfs_dict[method_name]
			color = detector_color_map.get(method_name, 'black')  # Default to black if method name not found in color map
			# print('Color for method', method_name, color)
			contribution_df = contribution_dfs_dict[method_name]
			customdata = []
			ranking_customdata = []
			# print('Contribution.shape', contribution_df.shape)
			# print('Score shape', scores_df.shape)
			print(f'Contribution df shape', contribution_df.shape)
			for row_index, row in contribution_df.iterrows():
				# print(f'Row index {row_index}', row)
				dimensional_label = multivariate_labels_df.iloc[row_index].values
				top_1_id = int(row[2])
				top_2_id = int(row[1])
				hit_k_score = row[0]
				# print(row_index, row)
				# str_list = f'<b>Score:{hit_k_score:.2f}--s{top_1_id}:s{top_2_id}</b>'
				str_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>:{f:.2f}' for i,f in enumerate(row.values.tolist())])
				ranking_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>' for i in (-row).argsort().values.tolist()])
				customdata.append(str_list)
				ranking_customdata.append(ranking_list)
			# print('Contribution shape', len(customdata))
			# fig.add_trace(
			# 	go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
			# 			   mode='lines', name=f"{method_name} score"),
			# 	row=num_series + 1,
			# 	col=1
			# )
			data.append(go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
						   mode='lines', name=f"{method_name} score", xaxis='x', yaxis=f'y{num_series+1}',
								   # customdata=['a:1, b:2, c:3'] * len(scores_df),
								   customdata=customdata,
								   line=dict(color=color),
								   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
								   hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}",
								   legendgroup=method_name
								   ),)

			data.append(go.Scatter(x=scores_df.index.to_list(), y=ranking_scores_numpy.tolist(),
								   mode='lines', name=f"{method_name} NDCG@k", xaxis='x', yaxis=f'y{num_series + 2}',
								   # customdata=['a:1, b:2, c:3'] * len(scores_df),
								   customdata=ranking_customdata,
								   hovertemplate="%{y:.4f}<br><b>Ranking</b>: %{customdata}",
								   # hovertemplate="%{y:.4f}",
								   line=dict(color=color),
								   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
								   # hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}"
								   legendgroup=method_name
								   ), )

	layout = dict(
		# height=100 * (num_series + 1),
		height=70 * (num_series + 2),
		showlegend=True,
		hoversubplots="axis",
		title=dict(text=f'Multivariate Time Series of the selected batch: {batch_id}. Anomaly scores of detectors are shown in the last subplot, with their contribution to the score in the hover.'),
		hovermode="x unified",
		grid=dict(rows=num_series+2, columns=1),
		# yaxis=dict(title=df.columns[0]),
		# yaxis1=dict(title=df.columns[1])
		# yaxes=[dict(title=f, showgrid=True, zeroline=False, showline=True, ticks='outside', row=num_series+1, col=1) for f in df.columns],
		# use_container_width=True
	)


	fig = go.Figure(data=data, layout=layout)
	for i, col in enumerate(df.columns):
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		xref = 'x' if i == 0 else f'x{i+1}'
		yref = 'y' if i == 0 else f'y{i+1}'
		# for anomaly_index in anomaly_ts.index.to_list():
		anomaly_windows = process_anomaly_index_to_windows(multivariate_labels_df[col])
		for start, end in anomaly_windows:
			# fig.add_vrect(x=anomaly_ts.index.to_list()[0], line_dash='solid', line_color='red', opacity=0.2, xref='x',
			# 			  yref=yref)
			fig.add_vrect(x0=start, x1=end, fillcolor='red', line_color='red', opacity=0.2, xref='x', yref=yref)
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor='red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	for i, col in enumerate(df.columns):
		fig.update_layout(**{f'yaxis{i+1}': dict(title=col, showgrid=True, zeroline=False, showline=True, ticks='outside')})
	fig.update_layout(**{f'yaxis{df.shape[1]+1}': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
													   # tickangle=30
													   )})
	fig.update_layout(**{
		f'yaxis{df.shape[1] + 2}': dict(title='NDCG@k', showgrid=True, zeroline=False, showline=True, ticks='outside',
										# tickangle=30
										)})
	# fig['layout']['yaxis1']['title'] = f'Sensor1'
	# for i, col in enumerate(df.columns):
	# 	# fig['layout']['xaxis{}'.format(i)]['title'] = f'Sensor{i}'
	# 	fig['layout']['yaxis{}'.format(i if i>0 else '')]['title'] = f'Sensor{i}'
	# Update layout
	# fig.update_layout(
	# 	height=100 * (num_series+1),
	# 	title_text="Batch of Multivariate Time Series",
	# 	showlegend=True,
	# 	hovermode="x unified",
	# 	hoversubplots= "axis",
	# )

	# hist_data = [
	# 	rng(0).standard_normal(200) - 2,
	# 	rng(1).standard_normal(200),
	# 	rng(2).standard_normal(200) + 2,
	# ]
	# group_labels = ["Group 1", "Group 2", "Group 3"]
	#
	# fig = ff.create_distplot(
	# 	hist_data, group_labels, bin_size=[0.1, 0.25, 0.5]
	# )

	# st.plotly_chart(fig)

	# fig.write_html(f'html/{batch_id}_mts_vs_scores.html')  # Save the figure to a file
	# print(f"Saved interactive plot for batch {batch_id} at html/{batch_id}_mts_vs_scores.html")
	fig.write_image(f'images/{batch_id}_mts_vs_scores.png', scale=1.0)  # Save the figure to a file
	print(f"Saved static image for batch {batch_id} at images/{batch_id}_mts_vs_scores.png")

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True, key='plot_mts')

def get_plot_batch_mts(batch_id, df, multivariate_labels_df, scores_dfs_dict, contribution_dfs_dict,
				   ranking_scores_dfs_dict,
				   detector_color_map, visualize_config):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create subplots for each time series
	num_series = len(df.columns)

	# fig = make_subplots(rows=num_series + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

	data = []
	# Add traces for each time series
	for i, col in enumerate(df.columns):
		# print(i, col)
		# print(df.shape)

		# fig.add_trace(
		# 	go.Scatter(x=df.index.to_list(), y=df[col].to_list(), mode='lines', name=col),
		# 	row=i + 1,
		# 	col=1
		# )

		data.append(go.Scatter(x=df.index.to_list(), y=df[col].to_list(),
							   mode='lines',
							   name=col,
							   xaxis='x',
							   yaxis='y' if i == 0 else f'y{i+1}',
							   legendgroup='mts_data',
							   # visible='legendonly',
							   ),)
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor= 'red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	if len(scores_dfs_dict.keys()) > 0:
		# Add traces for each score DataFrame
		for method_name, scores_df in scores_dfs_dict.items():
			ranking_scores_numpy = ranking_scores_dfs_dict[method_name]
			color = detector_color_map.get(method_name, 'black')  # Default to black if method name not found in color map
			# print('Color for method', method_name, color)
			contribution_df = contribution_dfs_dict[method_name]
			customdata = []
			ranking_customdata = []
			# print('Contribution.shape', contribution_df.shape)
			# print('Score shape', scores_df.shape)
			print(f'Contribution df shape', contribution_df.shape)
			for row_index, row in contribution_df.iterrows():
				# print(f'Row index {row_index}', row)
				dimensional_label = multivariate_labels_df.iloc[row_index].values
				top_1_id = int(row[2])
				top_2_id = int(row[1])
				hit_k_score = row[0]
				# print(row_index, row)
				# str_list = f'<b>Score:{hit_k_score:.2f}--s{top_1_id}:s{top_2_id}</b>'
				str_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>:{f:.2f}' for i,f in enumerate(row.values.tolist())])
				ranking_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>' for i in (-row).argsort().values.tolist()])
				customdata.append(str_list)
				ranking_customdata.append(ranking_list)
			# print('Contribution shape', len(customdata))
			# fig.add_trace(
			# 	go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
			# 			   mode='lines', name=f"{method_name} score"),
			# 	row=num_series + 1,
			# 	col=1
			# )
			if visualize_config.get('show_score', True):
				data.append(go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
							   mode='lines', name=f"{method_name} score", xaxis='x', yaxis=f'y{num_series+1}',
									   # customdata=['a:1, b:2, c:3'] * len(scores_df),
									   customdata=customdata,
									   line=dict(color=color),
									   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
									   hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}",
									   legendgroup=method_name
									   ),)
			if visualize_config.get('show_interpretability', True):
				data.append(go.Scatter(x=scores_df.index.to_list(), y=ranking_scores_numpy.tolist(),
									   mode='lines', name=f"{method_name} NDCG@k", xaxis='x', yaxis=f'y{num_series + 2}',
									   # customdata=['a:1, b:2, c:3'] * len(scores_df),
									   customdata=ranking_customdata,
									   hovertemplate="%{y:.4f}<br><b>Ranking</b>: %{customdata}",
									   # hovertemplate="%{y:.4f}",
									   line=dict(color=color),
									   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
									   # hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}"
									   legendgroup=method_name
									   ), )

	if visualize_config.get('show_score', True):
		num_series += 1
	if visualize_config.get('show_interpretability', True):
		num_series += 1

	layout = dict(
		# height=100 * (num_series + 1),
		height=70 * (num_series),
		showlegend=visualize_config.get('show_legend', True),
		hoversubplots="axis",
		title=dict(text=f'Multivariate Time Series of the selected batch: {batch_id}. Anomaly scores of detectors are shown in the last subplot, with their contribution to the score in the hover.') if visualize_config.get('show_title', True) else None,
		hovermode="x unified",
		grid=dict(rows=num_series, columns=1),
		# yaxis=dict(title=df.columns[0]),
		# yaxis1=dict(title=df.columns[1])
		# yaxes=[dict(title=f, showgrid=True, zeroline=False, showline=True, ticks='outside', row=num_series+1, col=1) for f in df.columns],
		# use_container_width=True
	)


	fig = go.Figure(data=data, layout=layout)
	for i, col in enumerate(df.columns):
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		xref = 'x' if i == 0 else f'x{i+1}'
		yref = 'y' if i == 0 else f'y{i+1}'
		# for anomaly_index in anomaly_ts.index.to_list():
		anomaly_windows = process_anomaly_index_to_windows(multivariate_labels_df[col])
		for start, end in anomaly_windows:
			# fig.add_vrect(x=anomaly_ts.index.to_list()[0], line_dash='solid', line_color='red', opacity=0.2, xref='x',
			# 			  yref=yref)
			fig.add_vrect(x0=start, x1=end, fillcolor='red', line_color='red', opacity=0.2, xref='x', yref=yref)
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor='red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	for i, col in enumerate(df.columns):
		fig.update_layout(**{f'yaxis{i+1}': dict(title=col, showgrid=True, zeroline=False, showline=True,
												 ticks='outside',
												 showticklabels=False,
												 side='left')})

	fig.update_layout(**{f'yaxis{df.shape[1]+1}': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
													   # tickangle=30
													   )})
	fig.update_layout(**{
		f'yaxis{df.shape[1] + 2}': dict(title='NDCG@k', showgrid=True, zeroline=False, showline=True, ticks='outside',
										# tickangle=30
										)})

	fig.update_layout(**{'plot_bgcolor': 'white'})
	# fig['layout']['yaxis1']['title'] = f'Sensor1'
	# for i, col in enumerate(df.columns):
	# 	# fig['layout']['xaxis{}'.format(i)]['title'] = f'Sensor{i}'
	# 	fig['layout']['yaxis{}'.format(i if i>0 else '')]['title'] = f'Sensor{i}'
	# Update layout
	# fig.update_layout(
	# 	height=100 * (num_series+1),
	# 	title_text="Batch of Multivariate Time Series",
	# 	showlegend=True,
	# 	hovermode="x unified",
	# 	hoversubplots= "axis",
	# )

	# hist_data = [
	# 	rng(0).standard_normal(200) - 2,
	# 	rng(1).standard_normal(200),
	# 	rng(2).standard_normal(200) + 2,
	# ]
	# group_labels = ["Group 1", "Group 2", "Group 3"]
	#
	# fig = ff.create_distplot(
	# 	hist_data, group_labels, bin_size=[0.1, 0.25, 0.5]
	# )

	# st.plotly_chart(fig)

	# fig.write_html(f'html/{batch_id}_mts_vs_scores.html')  # Save the figure to a file
	# print(f"Saved interactive plot for batch {batch_id} at html/{batch_id}_mts_vs_scores.html")
	# fig.write_image(f'images/{batch_id}_mts_vs_scores.png', scale=1.0)  # Save the figure to a file
	# print(f"Saved static image for batch {batch_id} at images/{batch_id}_mts_vs_scores.png")

	# Display the plot in Streamlit
	# st.plotly_chart(fig, use_container_width=True, key='plot_mts')
	return fig

def get_plot_batch_mts_and_table(batch_id, df, multivariate_labels_df, scores_dfs_dict, contribution_dfs_dict,
				   ranking_scores_dfs_dict,
				   detector_color_map, visualize_config, data_df):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create subplots for each time series
	num_series = len(df.columns)

	# fig = make_subplots(rows=num_series + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

	# data = []
	# Add traces for each time series
	fig = make_subplots(
		rows=1, cols=2,
		# shared_xaxes=True,
		# vertical_spacing=0.03,
		specs=[
			[{"type": "scatter"}, {"type": "table"}],
		]
	)
	for i, col in enumerate(df.columns):
		# print(i, col)
		# print(df.shape)

		# fig.add_trace(
		# 	go.Scatter(x=df.index.to_list(), y=df[col].to_list(), mode='lines', name=col),
		# 	row=i + 1,
		# 	col=1
		# )

		fig.add_trace(go.Scatter(x=df.index.to_list(), y=df[col].to_list(),
							   mode='lines',
							   name=col,
							   xaxis='x',
							   yaxis='y' if i == 0 else f'y{i+1}',
							   legendgroup='mts_data',
							   # visible='legendonly',
							   ), row=1, col=1)
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor= 'red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	if len(scores_dfs_dict.keys()) > 0:
		# Add traces for each score DataFrame
		for method_name, scores_df in scores_dfs_dict.items():
			ranking_scores_numpy = ranking_scores_dfs_dict[method_name]
			color = detector_color_map.get(method_name, 'black')  # Default to black if method name not found in color map
			# print('Color for method', method_name, color)
			contribution_df = contribution_dfs_dict[method_name]
			customdata = []
			ranking_customdata = []
			# print('Contribution.shape', contribution_df.shape)
			# print('Score shape', scores_df.shape)
			print(f'Contribution df shape', contribution_df.shape)
			for row_index, row in contribution_df.iterrows():
				# print(f'Row index {row_index}', row)
				dimensional_label = multivariate_labels_df.iloc[row_index].values
				top_1_id = int(row[2])
				top_2_id = int(row[1])
				hit_k_score = row[0]
				# print(row_index, row)
				# str_list = f'<b>Score:{hit_k_score:.2f}--s{top_1_id}:s{top_2_id}</b>'
				str_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>:{f:.2f}' for i,f in enumerate(row.values.tolist())])
				ranking_list = ",".join([f'<b style="color:{"red" if dimensional_label[i] == 1 else "black"};">s{i}</b>' for i in (-row).argsort().values.tolist()])
				customdata.append(str_list)
				ranking_customdata.append(ranking_list)
			# print('Contribution shape', len(customdata))
			# fig.add_trace(
			# 	go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
			# 			   mode='lines', name=f"{method_name} score"),
			# 	row=num_series + 1,
			# 	col=1
			# )
			if visualize_config.get('show_score', True):
				fig.add_trace(go.Scatter(x=scores_df.index.to_list(), y=scores_df[scores_df.columns[0]].to_list(),
							   mode='lines', name=f"{method_name} score", xaxis='x', yaxis=f'y{num_series+1}',
									   # customdata=['a:1, b:2, c:3'] * len(scores_df),
									   customdata=customdata,
									   line=dict(color=color),
									   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
									   hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}",
									   legendgroup=method_name
									   ),
							  row=1, col=1
							  )
			if visualize_config.get('show_interpretability', True):
				fig.add_trace(go.Scatter(x=scores_df.index.to_list(), y=ranking_scores_numpy.tolist(),
									   mode='lines', name=f"{method_name} NDCG@k", xaxis='x', yaxis=f'y{num_series + 2}',
									   # customdata=['a:1, b:2, c:3'] * len(scores_df),
									   customdata=ranking_customdata,
									   hovertemplate="%{y:.4f}<br><b>Ranking</b>: %{customdata}",
									   # hovertemplate="%{y:.4f}",
									   line=dict(color=color),
									   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
									   # hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}"
									   legendgroup=method_name
									   ),
							  row=1, col=1
							  )

	if visualize_config.get('show_score', True):
		num_series += 1
	if visualize_config.get('show_interpretability', True):
		num_series += 1

	layout = dict(
		# height=100 * (num_series + 1),
		height=70 * (num_series),
		showlegend=visualize_config.get('show_legend', True),
		hoversubplots="axis",
		title=dict(text=f'Multivariate Time Series of the selected batch: {batch_id}. Anomaly scores of detectors are shown in the last subplot, with their contribution to the score in the hover.') if visualize_config.get('show_title', True) else None,
		hovermode="x unified",
		grid=dict(rows=num_series, columns=1),
		# yaxis=dict(title=df.columns[0]),
		# yaxis1=dict(title=df.columns[1])
		# yaxes=[dict(title=f, showgrid=True, zeroline=False, showline=True, ticks='outside', row=num_series+1, col=1) for f in df.columns],
		# use_container_width=True
	)

	fig.update_layout(**layout)


	# fig = go.Figure(data=data, layout=layout)


	for i, col in enumerate(df.columns):
		# anomaly_ts = multivariate_labels_df[multivariate_labels_df[col] == 1.0][col]
		xref = 'x' if i == 0 else f'x{i+1}'
		yref = 'y' if i == 0 else f'y{i+1}'
		# for anomaly_index in anomaly_ts.index.to_list():
		anomaly_windows = process_anomaly_index_to_windows(multivariate_labels_df[col])
		for start, end in anomaly_windows:
			# fig.add_vrect(x=anomaly_ts.index.to_list()[0], line_dash='solid', line_color='red', opacity=0.2, xref='x',
			# 			  yref=yref)
			# fig.add_vrect(x0=start, x1=end, fillcolor='red', line_color='red', opacity=0.2, xref='x', yref=yref)
			fig.add_vrect(x0=start, x1=end, fillcolor='red', line_color='red', opacity=0.2, row=1, col=1)
		# data.append(go.Scatter(x=anomaly_ts.index.to_list(), y=df[col][anomaly_ts.index].to_list(),
		# 					   mode='markers',
		# 					   fillcolor='red',
		# 					   name=col,
		# 					   xaxis='x',
		# 					   yaxis='y' if i == 0 else f'y{i + 1}'
		# 					   ), )

	for i, col in enumerate(df.columns):
		fig.update_layout(**{f'yaxis{i+1}': dict(title=col, showgrid=True, zeroline=False, showline=True,
												 ticks='outside',
												 showticklabels=False,
												 side='left')})

	fig.update_layout(**{f'yaxis{df.shape[1]+1}': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
													   # tickangle=30
													   )})
	fig.update_layout(**{
		f'yaxis{df.shape[1] + 2}': dict(title='NDCG@k', showgrid=True, zeroline=False, showline=True, ticks='outside',
										# tickangle=30
										)})

	fig.update_layout(**{'plot_bgcolor': 'white'})

	fig.add_trace(
		go.Table(
			header=dict(
				values=[data_df.columns.to_list()],
				font=dict(size=10),
				align="left"
			),
			cells=dict(
				values=[data_df[k].tolist() for k in data_df.columns[0:]],
				align="left")
		),
		row=1, col=2
	)
	# fig['layout']['yaxis1']['title'] = f'Sensor1'
	# for i, col in enumerate(df.columns):
	# 	# fig['layout']['xaxis{}'.format(i)]['title'] = f'Sensor{i}'
	# 	fig['layout']['yaxis{}'.format(i if i>0 else '')]['title'] = f'Sensor{i}'
	# Update layout
	# fig.update_layout(
	# 	height=100 * (num_series+1),
	# 	title_text="Batch of Multivariate Time Series",
	# 	showlegend=True,
	# 	hovermode="x unified",
	# 	hoversubplots= "axis",
	# )

	# hist_data = [
	# 	rng(0).standard_normal(200) - 2,
	# 	rng(1).standard_normal(200),
	# 	rng(2).standard_normal(200) + 2,
	# ]
	# group_labels = ["Group 1", "Group 2", "Group 3"]
	#
	# fig = ff.create_distplot(
	# 	hist_data, group_labels, bin_size=[0.1, 0.25, 0.5]
	# )

	# st.plotly_chart(fig)

	# fig.write_html(f'html/{batch_id}_mts_vs_scores.html')  # Save the figure to a file
	# print(f"Saved interactive plot for batch {batch_id} at html/{batch_id}_mts_vs_scores.html")
	# fig.write_image(f'images/{batch_id}_mts_vs_scores.png', scale=1.0)  # Save the figure to a file
	# print(f"Saved static image for batch {batch_id} at images/{batch_id}_mts_vs_scores.png")

	# Display the plot in Streamlit
	# st.plotly_chart(fig, use_container_width=True, key='plot_mts')
	return fig

def plot_interpretability_curves(visualized_batch_id, combined_interpretability_metrics_of_base_detectors_df, detector_color_map):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""

	interpretability_detail_df = combined_interpretability_metrics_of_base_detectors_df[
		combined_interpretability_metrics_of_base_detectors_df['test_batch_id'] == visualized_batch_id]
	# Check if DataFrame is empty
	if interpretability_detail_df.empty:
		st.warning(f"👻 It's a ghost town in here... No data of batch {visualized_batch_id} found for plotting! Please select something from above to view data.")
		return

	vus_pr_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if
					  col.startswith('vus_pr')]
	L_value_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if
					   col not in vus_pr_columns and col.startswith('L_')]

	fig = go.Figure()

	# colors = list(mcolors.TABLEAU_COLORS.values())

	for i, alg in enumerate(old_method + ['avg_ens'] + best_ms_combination + best_ms_selection):
		color = detector_color_map.get(alg, 'black')  # Default to black if method name not found in color map

		alg_vus_pr_list = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			][vus_pr_columns].values.reshape(-1).tolist()

		alg_L_value_list = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			][L_value_columns].values.reshape(-1).tolist()

		# print(alg_vus_pr_list.shape, alg_L_value_list.shape)

		original_vus_pr = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			]['FFVUS_PR'].values.reshape(-1).tolist()[0]
		overall_interpretability_score = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			]['INTERPRETABILITY_SCORE'].values.reshape(-1).tolist()[0]

		# Interpretability line
		# print(alg_L_value_list[:10], alg_L_value_list[:10])
		fig.add_trace(go.Scatter(
			x=alg_L_value_list,
			y=alg_vus_pr_list,
			mode='lines+text',
			name=f'{alg}_VUSi (aVUSi={overall_interpretability_score:.3f})',
			line=dict(color=color),
			text=[f'{alg}_aVUSi={overall_interpretability_score:.3f}'],
			textposition='top right',
			textfont=dict(color=color),
			hovertemplate="%{y:.4f}",
			legendgroup=alg
		))
		# print(original_vus_pr*len(alg_L_value_list))
		fig.add_trace(go.Scatter(
			x=alg_L_value_list,
			y=[original_vus_pr] * len(alg_L_value_list),
			mode='lines+text',
			name=f'{alg}_VUS_PR={original_vus_pr:.3f}',
			line=dict(color=color, dash='dash'),
			text=[f'{alg}_VUS_PR={original_vus_pr:.3f}'],
			textposition='top right',
			textfont=dict(color=color),
			hovertemplate="%{y:.4f}",
			legendgroup=alg
		))

	# fig.add_trace(go.Scatter(
	#     x=alg_L_value_list,
	#     y=[overall_interpretability_score]*len(alg_L_value_list),
	#     mode='lines+text',
	#     name=f'{alg}_overall_interpretability={overall_interpretability_score:.3f}',
	#     line=dict(color=color, dash='dash'),
	#     text=[f'{alg}_overall_interpretability={overall_interpretability_score:.3f}'],
	#     textposition='bottom right',
	#     textfont=dict(color=color),
	#     legendgroup=alg
	# ))

	# Horizontal baseline (axhline equivalent)
	# fig.add_hline(
	#     y=original_vus_pr,
	#     line=dict(color=color, dash='dash'),
	#     annotation_text=f'{alg}_vus_pr_origin={original_vus_pr:.3f}',
	#     annotation_position='right',
	#     legendgroup=alg
	# )

	fig.update_layout(
		# width=1000,
		height=700,
		legend=dict(orientation='v'),
		xaxis_title='m_value',
		yaxis_title='VUSi',
		hovermode='x unified',
	)

	# fig.update_layout(**{f'yaxis{df.shape[1]+1}': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
	# 												   # tickangle=30
	# 												   )})

	# fig.write_html(f'html/{visualized_batch_id}.zip_interpretability_curves.html')  # Save the figure to a file
	# print(f"Saved interpretability curves plot for batch {visualized_batch_id} at html/{visualized_batch_id}.zip_interpretability_curves.html")
	fig.write_image(f'images/{visualized_batch_id}_interpretability_curves.png')  # Save the figure to a file
	print(f"Saved interpretability curves image for batch {visualized_batch_id} at images/{visualized_batch_id}_interpretability_curves.png")
	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True, key='plot_interpretability_curves')

def get_plot_interpretability_curves(visualized_batch_id, combined_interpretability_metrics_of_base_detectors_df, detector_color_map):
	"""
	Plots a batch of multivariate time series using Plotly.

	Args:
		df (DataFrame): DataFrame containing multivariate time series data.
	"""

	interpretability_detail_df = combined_interpretability_metrics_of_base_detectors_df[
		combined_interpretability_metrics_of_base_detectors_df['test_batch_id'] == visualized_batch_id]
	# print(interpretability_detail_df.columns)
	# print(interpretability_detail_df.head())
	# Check if DataFrame is empty
	# if interpretability_detail_df.empty:
	# 	st.warning(f"👻 It's a ghost town in here... No data of batch {visualized_batch_id} found for plotting! Please select something from above to view data.")
	# 	return

	vus_pr_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if
					  col.startswith('vus_pr')]
	L_value_columns = [col for col in combined_interpretability_metrics_of_base_detectors_df.columns if
					   col not in vus_pr_columns and col.startswith('L_')]

	fig = go.Figure()

	# colors = list(mcolors.TABLEAU_COLORS.values())

	for i, alg in enumerate(old_method + ['avg_ens'] + best_ms_combination + best_ms_selection):
		color = detector_color_map.get(alg, 'black')  # Default to black if method name not found in color map

		alg_vus_pr_list = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			][vus_pr_columns].values.reshape(-1).tolist()

		alg_L_value_list = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			][L_value_columns].values.reshape(-1).tolist()

		# print(alg_vus_pr_list.shape, alg_L_value_list.shape)

		original_vus_pr = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			]['FFVUS_PR'].values.reshape(-1).tolist()[0]
		overall_interpretability_score = interpretability_detail_df[
			interpretability_detail_df['algorithm'] == alg
			]['INTERPRETABILITY_SCORE'].values.reshape(-1).tolist()[0]

		# Interpretability line
		# print(alg_L_value_list[:10], alg_L_value_list[:10])
		fig.add_trace(go.Scatter(
			x=alg_L_value_list,
			y=alg_vus_pr_list,
			mode='lines+text',
			name=f'{alg}_VUSi (aVUSi={overall_interpretability_score:.3f})',
			line=dict(color=color),
			text=[f'{alg}_aVUSi={overall_interpretability_score:.3f}'],
			textposition='top right',
			textfont=dict(color=color),
			hovertemplate="%{y:.4f}",
			legendgroup=alg
		))
		# print(original_vus_pr*len(alg_L_value_list))
		fig.add_trace(go.Scatter(
			x=alg_L_value_list,
			y=[original_vus_pr] * len(alg_L_value_list),
			mode='lines+text',
			name=f'{alg}_VUS_PR={original_vus_pr:.3f}',
			line=dict(color=color, dash='dash'),
			text=[f'{alg}_VUS_PR={original_vus_pr:.3f}'],
			textposition='top right',
			textfont=dict(color=color),
			hovertemplate="%{y:.4f}",
			legendgroup=alg
		))

	# fig.add_trace(go.Scatter(
	#     x=alg_L_value_list,
	#     y=[overall_interpretability_score]*len(alg_L_value_list),
	#     mode='lines+text',
	#     name=f'{alg}_overall_interpretability={overall_interpretability_score:.3f}',
	#     line=dict(color=color, dash='dash'),
	#     text=[f'{alg}_overall_interpretability={overall_interpretability_score:.3f}'],
	#     textposition='bottom right',
	#     textfont=dict(color=color),
	#     legendgroup=alg
	# ))

	# Horizontal baseline (axhline equivalent)
	# fig.add_hline(
	#     y=original_vus_pr,
	#     line=dict(color=color, dash='dash'),
	#     annotation_text=f'{alg}_vus_pr_origin={original_vus_pr:.3f}',
	#     annotation_position='right',
	#     legendgroup=alg
	# )

	fig.update_layout(
		# width=1000,
		height=700,
		legend=dict(orientation='v'),
		xaxis_title='m_value',
		yaxis_title='VUSi',
		hovermode='x unified',
	)

	return fig

	# fig.update_layout(**{f'yaxis{df.shape[1]+1}': dict(title='Scores', showgrid=True, zeroline=False, showline=True, ticks='outside',
	# 												   # tickangle=30
	# 												   )})

	# fig.write_html(f'html/{visualized_batch_id}.zip_interpretability_curves.html')  # Save the figure to a file
	# print(f"Saved interpretability curves plot for batch {visualized_batch_id} at html/{visualized_batch_id}.zip_interpretability_curves.html")
	# fig.write_image(f'images/{visualized_batch_id}_interpretability_curves.png')  # Save the figure to a file
	# print(f"Saved interpretability curves image for batch {visualized_batch_id} at images/{visualized_batch_id}_interpretability_curves.png")
	# # Display the plot in Streamlit
	# st.plotly_chart(fig, use_container_width=True, key='plot_interpretability_curves')

def generate_dataframe(df, datasets, methods_family, length, type_exp='_score'):
	"""
	Generates a DataFrame filtered by dataset, method families, and lengths, for different types of experiments.

	Args:
		df (DataFrame): Original DataFrame containing data to be filtered.
		datasets (list): List of datasets to include in the filtered DataFrame.
		methods_family (list): List of method families to include in the filtered DataFrame.
		length (list): List of lengths to include in the filtered DataFrame.
		type_exp (str, optional): Type of experiment ('_score', '_inf', or '_time'). Defaults to '_score'.

	Returns:
		DataFrame: Filtered DataFrame based on the specified parameters.

	Example:
		filtered_df = generate_dataframe(df, ['dataset1', 'dataset2'], ['family1', 'family2'], [10, 20], '_inf')
	"""

	# Filter DataFrame based on type of experiment
	if type_exp == '_score':
		# Generate DataFrame for scoring experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', type_exp)
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length] + old_method + baselines + best_ms_combination + best_ms_selection]
	elif type_exp == '_inf':
		# Generate DataFrame for inference experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', type_exp)
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length]]
	elif type_exp == '_time':
		# Generate DataFrame for time-related experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', '').replace('_default', '')
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length] + old_method + baselines + best_ms_combination + best_ms_selection]



def add_rect(label, data):
	"""
	Highlights anomalies in a time series plot.

	Args:
		label (list): List of labels indicating anomalies (1 for anomaly, 0 for normal).
		data (list): List of data points representing the time series.

	Returns:
		list: List containing highlighted anomalies in the time series plot.

	Example:
		highlighted_ts = add_rect([0, 1, 0, 0, 1], [1, 2, 3, 4, 5])
	"""

	# Initialize list for plotting anomalies
	anom_plt = [None] * len(data)

	# Create a copy of the original data
	ts_plt = data.copy()

	# Get the length of the time series
	len_ts = len(data)

	# Loop through labels and data to identify anomalies
	for i, lab in enumerate(label):
		if lab == 1:
			# Mark anomalies and neighboring points
			anom_plt[i] = data[i]
			anom_plt[min(len_ts - 1, i + 1)] = data[min(len_ts - 1, i + 1)]

	return anom_plt


def rename_columns(df):
	
	# Function to process each column name
	def process_column_name(col_name):
		for length in list_length:
			for template, nice_name in template_names.items():
				if col_name.startswith(template.format(length)):
					return col_name.replace(template.format(length), nice_name.format(length))
		# If no template matches, return the original column name
		return col_name

	# Rename columns using the function
	new_columns = {col: process_column_name(col) for col in df.columns}
	df.rename(columns=new_columns, inplace=True)
	return df


def rename_columns_and_update_palette(df, palette):
	# List of lengths used in renaming
	list_length = [16, 32, 64, 128, 256, 512, 768, 1024]

	# Function to process each column name and update the palette
	def process_column_name(col_name):
		for length in list_length:
			for template, nice_name in template_names.items():
				formatted_template = template.format(length)
				if col_name.startswith(formatted_template):
					new_name = col_name.replace(formatted_template, nice_name.format(length))
					if col_name in palette:
						palette[new_name] = palette.pop(col_name)
					return new_name
		return col_name  # Return original name if no template matches

	# Update columns and palette
	new_columns = {col: process_column_name(col) for col in df.columns}
	df.rename(columns=new_columns, inplace=True)

	return df, palette

def estimate_dimension_contribution_with_a_buffer(dimension_contribution: np.ndarray, buffer: int) -> np.array:
	"""
    Estimate the contribution of each dimension with a buffer.

    Args:
        dimension_contribution: 2D array of shape (T, D) with contribution scores for each dimension at each timestamp.
        buffer: number of timestamps to consider before and after the detected anomaly.

    Returns:
        2D array of shape (T, D) with estimated contribution scores for each dimension at each timestamp.
    """

	T, D = dimension_contribution.shape
	estimated_contribution = np.zeros_like(dimension_contribution)

	for t in range(T):
		start = max(0, t - buffer)
		end = min(T, t + buffer + 1)
		if np.version.version >= '2.0.0':
			estimated_contribution[t] = np.trapezoid(dimension_contribution[start:end, :], axis=0)
		else:
			estimated_contribution[t] = np.trapz(dimension_contribution[start:end, :], axis=0)
	return estimated_contribution

def set_streamlit_page_config_once(mode):
	try:
		streamlit.set_page_config(layout=mode)
	except streamlit.errors.StreamlitAPIException as e:
		if "can only be called once per app" in e.__str__():
			return
		raise e