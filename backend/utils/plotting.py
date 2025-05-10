import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def generate_plot(df: pd.DataFrame, chart_type: str, x_column: Optional[str] = None, 
                 y_column: Optional[str] = None, title: Optional[str] = None) -> Optional[str]:
    """Generates a plot and returns a base64 encoded image string."""
    if df.empty:
        return None
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", 8)
    plt.clf()  # Clear any previous plots

    # Auto-detect columns if not provided
    if x_column is None and len(df.columns) > 0:
        x_column = df.columns[0]
    if y_column is None:
        if len(df.columns) > 1:
            y_column = df.columns[1]
        elif len(df.columns) == 1:  # Use index for x if only one column for y
            y_column = df.columns[0]
            df = df.reset_index()
            x_column = 'index'
        else:  # No columns to plot
            return None

    try:
        if chart_type == 'bar':
            sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette, alpha=0.8)
        elif chart_type == 'line':
            sns.lineplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                markers=True,  # Add markers at data points
                dashes=False,  # Solid lines
                marker='o',    # Circle markers
                markersize=8,  # Marker size
                markeredgecolor='white',  # White edge for contrast
                markeredgewidth=1.5,      # Edge width
                linewidth=2.5,            # Line thickness
                color=palette[0]          # Line color
            )
            plt.grid(True, linestyle='--', alpha=0.7)
        elif chart_type == 'scatter':
            size_col = df[y_column] if pd.api.types.is_numeric_dtype(df[y_column]) else None
            
            sns.scatterplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                hue=df.columns[2] if len(df.columns) > 2 else None,  # Use third column for color if available
                size=size_col,  # Dynamic sizing based on y-value
                sizes=(50, 200),  # Min and max point size
                alpha=0.7,  # Transparency
                palette=palette,
                edgecolor='white',  # White edges for contrast
                linewidth=0.5
            )
            if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
                sns.regplot(x=df[x_column], y=df[y_column], scatter=False, line_kws={"color": "red", "alpha": 0.7, "lw": 2, "ls": "--"})
                
        elif chart_type == 'box':
            sns.boxplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                palette=palette,
                width=0.6,  # Box width
                fliersize=5,  # Outlier point size
                linewidth=1.5  # Line width for boxes
            )
            # Add strip plot on top of boxplot for actual data distribution
            sns.stripplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                size=4, 
                color='black', 
                alpha=0.5
            )
        elif chart_type == 'pie' and len(df) <= 10:
            # Ensure y_column is numeric for pie chart values
            if pd.api.types.is_numeric_dtype(df[y_column]):
                 plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%', startangle=90)
            else:  # Fallback or error if y_column is not numeric
                sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette)  # Fallback to bar
                chart_type = 'bar (fallback from pie due to non-numeric y-axis)'
        elif chart_type == 'hist':
            sns.histplot(
                df[y_column if y_column in df.columns else x_column],
                kde=True,
                color=palette[0],
                alpha=0.7,
                edgecolor='white',
                linewidth=1
            )
            # Add mean line
            mean_val = df[y_column if y_column in df.columns else x_column].mean()
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                        label=f'Mean: {mean_val:.2f}')
            plt.legend()
        else:  # Default to bar chart
            ax = sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette)
            chart_type = f'bar (defaulted from {chart_type})'

        plt.title(title if title else f"{chart_type.capitalize()} of {y_column} by {x_column}")
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(x_column, fontsize=12, labelpad=10)
        plt.ylabel(y_column, fontsize=12, labelpad=10)
        plt.xticks(rotation=45 if len(df) > 5 else 0, ha='right' if len(df) > 5 else 'center')
        plt.tight_layout()
        plt.figtext(0.9, 0.05, 'Data Insights', fontstyle='italic', alpha=0.5)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return plot_base64
    except Exception as e:
        print(f"Error generating plot: {e}")
        plt.close()
        return None 