"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numpy.random import default_rng as rng
import plotly.figure_factory as ff
import numpy as np

from .constant import methods_colors, methods_conv, methods_sit, methods_ts, methods_classical, old_method, list_length, method_group, template_names


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
	for family, color in zip([methods_conv, methods_sit, methods_ts, methods_classical],
							 [methods_colors["conv"], methods_colors["sit"], methods_colors["rocket"],
							  methods_colors["feature_based"]]):
		for length in list_length:
			my_pal_tmp = {method.format(length): color for method in family}
			my_pal = {**my_pal, **my_pal_tmp}
	my_pal.update({"Avg Ens": methods_colors["avg_ens"], 'Oracle': methods_colors["oracle"]})

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
		orient='h',
		saturation=1, 
		whis=0.241289844
	)
	
	g.xaxis.grid(True)
	plt.xlabel(measure_name)
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

def plot_batch_mts(df, multivariate_labels_df, scores_dfs_dict, contribution_dfs_dict):
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
							   yaxis='y' if i == 0 else f'y{i+1}'
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
			contribution_df = contribution_dfs_dict[method_name]
			customdata = []
			# print('Contribution.shape', contribution_df.shape)
			# print('Score shape', scores_df.shape)
			print(f'Contribution df shape', contribution_df.shape)
			for row_index, row in contribution_df.iterrows():
				# print(f'Row index {row_index}', row)
				top_1_id = int(row[2])
				top_2_id = int(row[1])
				hit_k_score = row[0]
				# print(row_index, row)
				# str_list = f'<b>Score:{hit_k_score:.2f}--s{top_1_id}:s{top_2_id}</b>'
				str_list = ",".join([f'<b>s{i}</b>:{f:.2f}' for i,f in enumerate(row.values.tolist())])
				customdata.append(str_list)
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
								   # hovertemplate="%{y:.4f}<br><b>Interpretability Hit@2</b>: %{customdata}"
								   hovertemplate="%{y:.4f}<br><b>Contribution</b>: %{customdata}"
								   ),)

	layout = dict(
		# height=100 * (num_series + 1),
		height=70 * (num_series + 1),
		showlegend=True,
		hoversubplots="axis",
		title=dict(text=f'Multivariate Time Series of the selected batch'),
		hovermode="x unified",
		grid=dict(rows=num_series+1, columns=1),
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

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True)


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
													  for l in length] + old_method]
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
													  for l in length] + old_method]



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
