import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Example data
beam_sizes = [1, 5, 10, 15, 18, 20, 23, 25]  # Replace with your beam sizes
bleu_scores = [
    15.9,
    19.3,
    20.8,
    21.4,
    21.6,
    21.7,
    21.2,
    21.0,
]  # Replace with BLEU scores
brevity_penalties = [
    1.000,
    1.000,
    1.000,
    1.000,
    0.992,
    0.977,
    0.932,
    0.899,
]  # Replace with brevity penalties

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=beam_sizes, y=bleu_scores, name="BLEU Score",
               mode='lines+markers',
               line=dict(color='#2E86C1', width=3),
               marker=dict(size=8, symbol='circle')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=beam_sizes, y=brevity_penalties, name="Brevity Penalty",
               mode='lines+markers',
               line=dict(color='#E74C3C', width=3, dash='dash'),
               marker=dict(size=8, symbol='x')),
    secondary_y=True,
)

# Update layout
fig.update_layout(
    title={
        'text': "BLEU Score and Brevity Penalty vs Beam Size",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=32)
    },
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        font=dict(size=24)
    )
)

# Update axes
fig.update_xaxes(title_text="Beam Size", gridcolor='lightgray', title_font=dict(size=24), tickfont=dict(size=20))
fig.update_yaxes(title_text="BLEU Score", secondary_y=False, 
                 gridcolor='lightgray', color='#2E86C1', title_font=dict(size=24), tickfont=dict(size=20))
fig.update_yaxes(title_text="Brevity Penalty", secondary_y=True, 
                 gridcolor='lightgray', color='#E74C3C', title_font=dict(size=24), tickfont=dict(size=20))

# Save and show the plot
fig.write_html("bleu_vs_brevity_penalty.html")  # Interactive HTML file
fig.write_image("bleu_vs_brevity_penalty.png", 
                width=1920, height=1080,  # Full HD resolution
                scale=2)  # 2x scaling for higher quality
fig.show()
