import pandas as pd
import streamlit as st

def plot_summary(df, y, percentage=False):
    df = df.copy()
    
    x = "Epoch"
    c = "Optimizer"
    
    sub_title = f"Lower is better"
    range_y = None
    # if y == "Generalization_Gap":
    # 	sub_title = f"Lower is better"
    # 	range_y = None
    # else:
    # 	range_y = [0, 100 if percentage else 1]
    # 	sub_title = f"Higher is better"

    if percentage:
        df[y] *= 100

    title = f'{y.replace("_", " ")}'

    title += f"<br><sup>{sub_title}</sup>"

    fig = px.line(
        data_frame=df,
        x=x,
        y=y,
        color = c,
        title = title,
        range_x = [df[x].values.min(), df[x].values.max()],
        range_y = range_y, # df[y].values.min() * 0.95
        markers=True,
    )
 
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
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
 
    return fig

@st.cache_data(ttl=60*60)
def get_summary():
    return pd.read_csv("summary.csv")

def agg(summary):
    summary = (
        summary
        .groupby(["Epoch", "Subset"])
        .agg(["mean"])
    )
    summary.columns = list(map('_'.join, summary.columns.values))
    summary = (
        summary
        .reset_index()
        .pivot(
            index="Epoch",
            columns="Subset",
            # values = "Accuracy"
        )
    )
    summary.columns = list(map('_'.join, summary.columns.values))
    summary["Generalization_Gap"] = summary["Loss_mean_Dev"] - summary["Loss_mean_Train"]
    summary = summary.reset_index()
    
    return summary

def main():
    summary = get_summary()
    
    plot_summary(
        summary,
        "Loss_mean_Train"
    )
    # plot_summary(
    #     summary,
    #     "Loss_mean_Dev"
    # )
    # plot_summary(
    #     summary,
    #     "Generalization_Gap"
    # )
    return

if __name__ == "__main__":
    main()
    
