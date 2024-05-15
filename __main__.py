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

def main():
    summary = get_summary()
    
    menu_options = [
        "Loss_mean_Train",
        "Loss_mean_Dev",
        "Generalization_Gap",
    ]
    
    with st.sidebar:
        menu_selected = st.radio(
            "Visualization",
            menu_options
        )
    
    for option in menu_options:
        if menu_selected == option:
            if menu_selected == "Generalization_Gap":
                summary["Generalization_Gap"] = summary["Loss_mean_Dev"] - summary["Loss_mean_Train"]
                
            plot_summary(
                summary,
                option
            )
    
    return

if __name__ == "__main__":
    main()
    
