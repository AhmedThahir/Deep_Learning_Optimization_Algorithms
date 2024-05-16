import pandas as pd
import streamlit as st

import plotly.express as px

@st.cache_data(ttl=60*60)
def get_summary():
	df = pd.read_csv("summary.csv")
	df["Loss_Generalization_Gap"] = df["Loss_mean_Dev"] - df["Loss_mean_Train"]
	df["Accuracy_Generalization_Gap"] = df["Accuracy_mean_Train"] - df["Accuracy_mean_Dev"]
	return df

@st.cache_data(ttl=60*60)
def filter_optimizers(df, optimizers_selected):
	return df[
		df["Optimizer"].isin(optimizers_selected)
	]

def plot_summary(df, x, y):
	df = df.copy()
	c = "Optimizer"
	
	if "Accuracy" in y and "Generalization" not in y:
		sub_title = f"Higher is better"
		percentage = True
	else:
		sub_title = f"Lower is better"
		percentage = False
	
	if percentage:
		df[y] *= 100

	if "Accuracy" in y and "Generalization" not in y:
		range_y = [
			0,
			100
		]
	else:
		range_y = [
			0,
			df[
				df[y] > 0
			][y].quantile(0.90)*1.1
		]

	# if "loss" in y.lower():
	# 	range_y = [0, df[y].quantile(0.90)*1.1]
	# else:
	# 	range_y = None
	# if y == "Generalization_Gap":
	# 	sub_title = f"Lower is better"
	# 	range_y = None
	# else:
	# 	range_y = [0, 100 if percentage else 1]
	# 	sub_title = f"Higher is better"

	title = f'{y.replace("_", " ")}'

	title += f"<br><sup>{sub_title}</sup>"
	
	facet_row = "Train_Batch_Size"

	fig = px.line(
		data_frame=df,
		x=x,
		y=y,
		facet_col="Learning_Rate",
		facet_row="Train_Batch_Size",
		facet_row_spacing = 0.1,
		color = c,
		title = title,
		range_x = [df[x].values.min(), df[x].values.max()],
		range_y = range_y, # df[y].values.min() * 0.95
		markers=True,
	)
	
	n_rows = df[facet_row].unique().shape[0]
	fig.update_layout(height=300*n_rows)
	fig.update_traces(
		patch={
			"marker": {"size": 5},
			"line": {
				"width": 1,
				# "dash": "dot"
			},
		}
	)
	fig.update_traces(connectgaps=True) # required for connecting dev accuracies
	st.plotly_chart(fig, use_container_width=True)
 
	return fig

def main():
	st.set_page_config(
		layout = "wide"
	)
	summary = get_summary()
	
	x_options = [
		"Epoch",
		"Train_Time",
	]
 
	exclude = [
		"Model",
  		"Optimizer",
		"Learning_Rate",
		"Train_Batch_Size"
	]
	
	y_options = [col for col in summary.columns if col not in x_options and col not in exclude]
	optimizers_options = list(summary["Optimizer"].unique())
	
	with st.sidebar:
		x_selected = st.radio(
			"X Metric",
			x_options
		)
		y_selected = st.radio(
			"Y Metric",
			y_options
		)
		optimizers_selected = st.multiselect(
			"Optimizers",
			optimizers_options,
			# value = ["Adam", "SGD"]
		)
		
	if len(optimizers_selected) == 0:
		optimizers_selected = optimizers_options
		# st.warning("Select optimizer(s)")
		# return
	
	summary = summary.pipe(filter_optimizers, optimizers_selected)
		
	plot_summary(
		summary,
		x=x_selected,
		y=y_selected
	)
	
	return

if __name__ == "__main__":
	main()
	
