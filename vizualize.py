import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import torch


def get_plot3D(surface, title):
    h_size = 600
    font_size = 18
    title_font_size = 30

    x = np.linspace(0, 1, surface.shape[0])
    y = np.linspace(0, 1, surface.shape[0])
    fig = go.Figure(
        data=[
            go.Surface(z=surface, x=x, y=y),
        ]
    ).update_layout(
        title=title,
        autosize=False,
        # margin=dict(l=65, r=50, b=65, t=90),
        height=h_size,
        width=h_size,
        # scene_aspectmode="cube",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(family="Times New Roman,italic", size=font_size),
        # scene_camera=dict(eye=dict(x=2.5, y=0, z=1)),
        scene={
            "aspectratio": {"x": 1, "y": 1, "z": 0.5},
        },
        # template="seaborn",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="<i>x</i>",
            yaxis_title="<i>y</i>",
            zaxis_title="<i>z</i>",
            xaxis_title_font_size=title_font_size,
            yaxis_title_font_size=title_font_size,
            zaxis_title_font_size=title_font_size,
            zaxis=dict(
                # tickvals=np.linspace(0.1, 0.9, 5),
                # ticktext=["0.3", "0.6", "<i>z</i>"],
                # backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                nticks=6,
                tickwidth=6,
                zerolinecolor="white",
                ticklen=10,
                ticks="outside",
            ),
            yaxis=dict(
                tickvals=np.linspace(0.1, 0.9, 5),
                ticklen=5,
                tickwidth=6,
                ticks="outside",
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            xaxis=dict(
                tickvals=np.linspace(0.1, 0.9, 5),
                ticklen=5,
                tickwidth=6,
                ticks="outside",
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
        )
    )

    fig.update_traces(
        selector=dict(type="surface"),
    )

    fig.update_traces(showscale=False)
    return fig


def get_heatmap(pred, percent=True):
    line_size = 10

    text_size = 50
    h_size = 1000
    w_size = h_size * 1.3
    font_size = 18

    x = np.linspace(0, 1, pred.shape[0])
    y = np.linspace(0, 1, pred.shape[0])

    fig = go.Figure(
        data=go.Heatmap(
            x=np.sort(x), y=np.sort(y), z=pred, type="heatmap", colorscale="Viridis"
        )
    )
    fig.update_layout(
        autosize=False,
        width=w_size,
        height=h_size,
        font=dict(family="Times New Roman", size=text_size, color="black"),
        margin=dict(l=0, r=30, b=0, t=30, pad=0),
    )

    fig.update_traces(
        cauto=False,
        selector=dict(type="surface"),
        colorbar=dict(
            lenmode="fraction",
            len=0.4,
            thickness=15,
            tickfont=dict(size=font_size / 2),
            nticks=5,
        ),
    )

    # %% Set xandy-axis
    fig.update_xaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        title_text=f"<i>x</i>",
        title_font_size=text_size,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        title_text=f"<i>y</i>",
        title_font_size=text_size,
        title_standoff=50,
    )

    return fig


def get_error_heatmap(pred, true, percent=True):
    line_size = 10

    text_size = 50
    h_size = 1000
    w_size = h_size * 1.3
    font_size = 18

    x = np.linspace(0, 1, pred.shape[0])
    y = np.linspace(0, 1, pred.shape[0])
    if percent:
        error = np.abs((pred - true))
        error = error * 100 / torch.max(error)
    else:
        error = np.abs((pred - true))

    fig = go.Figure(
        data=go.Heatmap(
            x=np.sort(x), y=np.sort(y), z=error, type="heatmap", colorscale="Viridis"
        )
    )
    fig.update_layout(
        autosize=False,
        width=w_size,
        height=h_size,
        font=dict(family="Times New Roman", size=text_size, color="black"),
        margin=dict(l=0, r=30, b=0, t=30, pad=0),
    )

    fig.update_traces(
        cauto=False,
        selector=dict(type="surface"),
        colorbar=dict(
            lenmode="fraction",
            len=0.4,
            thickness=15,
            tickfont=dict(size=font_size / 2),
            nticks=5,
        ),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        title_text=f"<i>x</i>",
        title_font_size=text_size,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        title_text=f"<i>y</i>",
        title_font_size=text_size,
        title_standoff=50,
    )

    return fig


def get_plot2D(pred_sliced, true_sliced, shape):
    fig = go.Figure()
    x = np.linspace(0, 1, shape)
    fig.add_trace(go.Scatter(x=x, y=pred_sliced, mode="lines", name="pred"))
    fig.add_trace(go.Scatter(x=x, y=true_sliced, mode="lines", name="true"))

    return fig


def get_full_plot2D(sliced, shape, cut_type):
    from plotly.graph_objects import Layout

# Set layout with background color you want (rgba values)
# This one is for white background
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    line_size = 7  # TikZ line width 1pt
    marker_size = 6  # TikZ marker size 2.5pt converted to Plotly scale
    text_size = 50  # Adjust text size for clarity in Plotly
    h_size = 1000
    w_size = h_size

    x = np.linspace(0, 1, shape)

    modes = ["solid", "dashdot", "longdash"] * 4

    traces = [(*sliced[i], modes[i]) for i in range(len(sliced))]

    fig = go.Figure(layout=layout)

    for trace, name, mode in traces:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=trace,
                mode="lines",
                name=name,
                line=dict(
                    width=line_size,
                    dash=mode,
                ),
                marker=dict(
                    size=marker_size,
                    symbol='circle'  # TikZ marker style
                )
            )
        )

    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=w_size,
        height=h_size,
        font=dict(family="Times New Roman", size=text_size, color="black"),
        margin=dict(l=0, r=30, b=0, t=30, pad=0),
        legend=dict(
            orientation="h",
            bordercolor="black",
            borderwidth=2,
            x=0.28,
            y=1.05,
            traceorder="normal",
            itemsizing="trace",
            itemwidth=30,
            font=dict(family="Times New Roman", size=3 * text_size / 4, color="black"),
        ),
    )
    # fig.update_layout(
    #     autosize=False,
    #     width=600,
    #     height=400,
    #     font=dict(family="Times New Roman", size=text_size, color="black"),
    #     margin=dict(l=40, r=40, b=40, t=40, pad=0),
    #     legend=dict(
    #         orientation="v",
    #         x=0.98,
    #         y=0.98,
    #         xanchor="right",
    #         yanchor="top",
    #         bordercolor="black",
    #         borderwidth=1,
    #         font=dict(family="Times New Roman", size=text_size, color="black"),
    #     ),
    # )
    
    # %% Set xandy-axis
    axis_text = "<i>x</i>" if cut_type.startswith("x") else "<i>y</i>" 
    fig.update_xaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=2,
        gridcolor="gray",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        title_text=axis_text,
        title_font_size=text_size,
        #  tickvals  = [0. + i*0.02 for i in range(5)],
        # ticktext  = [str(10*i) for i in range(6)]+['<i>t<i>'],
        # nticks=6
    )
    fig.update_yaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=2,
        gridcolor="gray",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="gray",
        title_text=f"<i>z</i>",
        title_font_size=text_size,
        title_standoff=50,
        # nticks=5,
        # range=[0,z_max[k]],
    )
    
    
    return fig


def get_errors_hist(errors):
    # errors = []
    # for batch in results:
    #     pred = batch[0]
    #     y = batch[1]

    #     for sample_pred, sample_y in zip(pred, y):
    #         errors.append(relative_error(sample_pred, sample_y))

    fig = px.histogram(errors, labels={"value": "relative error"})
    fig.update_layout(showlegend=False)
    return fig


def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


def get_boxplot(all_modules_errors):
    line_size = 10

    text_size = 20
    h_size = 1000
    w_size = h_size
    fig = go.Figure()
    for i, model in enumerate(all_modules_errors):
        errors = removeOutliers(all_modules_errors[model], 0.2)
        errors = all_modules_errors[model]
        fig.add_trace(go.Box(y=errors, name=f"({i+1})"))

    fig.update_layout(
        font=dict(family="Times New Roman", size=text_size, color="black"),
        # margin=dict(
        #     l=40,
        #     r=30,
        #     b=80,
        #     t=100,
        # ),
        showlegend=False,
        # margin=dict(l=0, r=30, b=0, t=30, pad=0),
        # legend=dict(
        #     orientation="h",
        #     bordercolor='black',
        #     borderwidth=2,
        #     x=0.28,
        #     y=1.05,
        #     traceorder="normal",
        #     itemsizing="trace",
        #     itemwidth=30,
        #     font=dict(family="Times New Roman", size=3 * text_size / 4, color="black"),
        # ),
        template="plotly_white",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="LightPink",
        title_font_size=text_size,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="LightPink",
        title_text=f"<i>Относительная ошибка</i>",
        title_font_size=text_size,
        title_standoff=30,
        # nticks=6,
        # range=[0,z_max[k]],
    )
    # fig.update_yaxes(nticks=6)

    return fig
