import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
from io import BytesIO

# Page config
st.set_page_config(page_title="Interactive Data Explorer", layout="wide")

st.title("📊 Interactive Data Explorer")
st.markdown("Upload a CSV file to auto-profile your data and explore relationships interactively.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display raw data (optional sample)
    if st.checkbox("Show sample data (first 5 rows)"):
        st.dataframe(df.head())
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a view:", ["Data Profile", "Missing Values", "Correlations", "Visualize Relationships"])
    
    if page == "Data Profile":
        st.header("🔍 Data Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Data Types & Missing Counts")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Missing Count': df.isnull().sum()
            })
            st.dataframe(info_df)
    
    elif page == "Missing Values":
        st.header("❌ Missing Values")
        fig_msno = msno.matrix(df)
        st.plotly_chart(fig_msno, use_container_width=True)
    
    elif page == "Correlations":
        st.header("🔗 Correlations")
        # Compute correlation matrix (numeric columns only)
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No numeric columns found for correlations.")
    
    elif page == "Visualize Relationships":
        st.header("📈 Visualize Relationships")
        
        # Select columns for scatter plot
        col_options = df.select_dtypes(include=['number']).columns.tolist()
        if col_options:
            x_col = st.selectbox("X-axis:", col_options)
            y_col = st.selectbox("Y-axis:", col_options, index=1 if len(col_options) > 1 else 0)
            
            # Interactive scatter plot
            fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                                     hover_data=df.columns, 
                                     title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Bonus: Add a dropdown for color-by column
            color_col = st.selectbox("Color by:", ["None"] + df.columns.tolist())
            if color_col != "None":
                fig_scatter = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                         hover_data=df.columns, 
                                         title=f"Scatter Plot: {x_col} vs {y_col} (colored by {color_col})")
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No numeric columns found for visualization.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Pandas, and Plotly. [GitHub Repo](https://github.com/yourusername/data-explorer) | Upload your own CSV to explore!")
