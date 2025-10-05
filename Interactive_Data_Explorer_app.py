import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Interactive Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Interactive Data Explorer")
st.markdown("Upload any CSV file to automatically profile and visualize your data!")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display success message
        st.success(f"✅ Successfully loaded: {uploaded_file.name}")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Preview", 
            "📊 Data Profile", 
            "📈 Visualizations", 
            "🔗 Correlations",
            "🔍 Filter Data"
        ])
        
        # TAB 1: Data Preview
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            st.subheader("Dataset Shape")
            col1, col2 = st.columns(2)
            col1.metric("Number of Rows", df.shape[0])
            col2.metric("Number of Columns", df.shape[1])
            
            st.subheader("Column Names and Types")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # TAB 2: Data Profile
        with tab2:
            st.subheader("📊 Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Duplicate Rows", df.duplicated().sum())
            col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Missing values
            st.subheader("Missing Values Analysis")
            missing = df.isnull().sum()
            missing_percent = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing Percentage': missing_percent.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualize missing data
                fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                            title='Missing Data by Column (%)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            # Numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                st.write("**Numerical Columns:**")
                st.dataframe(df[num_cols].describe(), use_container_width=True)
            
            # Categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                st.write("**Categorical Columns:**")
                cat_summary = pd.DataFrame({
                    'Column': cat_cols,
                    'Unique Values': [df[col].nunique() for col in cat_cols],
                    'Most Frequent': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in cat_cols],
                    'Frequency': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in cat_cols]
                })
                st.dataframe(cat_summary, use_container_width=True)
        
        # TAB 3: Visualizations
        with tab3:
            st.subheader("📈 Interactive Visualizations")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if num_cols:
                # Distribution plots
                st.write("**Distribution Plots (Numerical Data)**")
                selected_num_col = st.selectbox("Select a numerical column", num_cols, key='dist')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(df, x=selected_num_col, 
                                           title=f'Histogram: {selected_num_col}',
                                           marginal='box')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_num_col, 
                                    title=f'Box Plot: {selected_num_col}')
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Scatter plot
                if len(num_cols) >= 2:
                    st.write("**Scatter Plot (Compare Two Variables)**")
                    col1, col2 = st.columns(2)
                    x_axis = col1.selectbox("Select X-axis", num_cols, key='scatter_x')
                    y_axis = col2.selectbox("Select Y-axis", num_cols, index=1 if len(num_cols) > 1 else 0, key='scatter_y')
                    
                    color_option = st.selectbox("Color by (optional)", ['None'] + cat_cols, key='scatter_color')
                    
                    if color_option == 'None':
                        fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                                title=f'{x_axis} vs {y_axis}')
                    else:
                        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color_option,
                                                title=f'{x_axis} vs {y_axis}')
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            if cat_cols:
                st.write("**Categorical Data Distribution**")
                selected_cat_col = st.selectbox("Select a categorical column", cat_cols, key='cat')
                
                # Count plot
                value_counts = df[selected_cat_col].value_counts().head(20)
                fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                labels={'x': selected_cat_col, 'y': 'Count'},
                                title=f'Top 20 Categories in {selected_cat_col}')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart
                if len(value_counts) <= 10:
                    fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                                    title=f'Distribution of {selected_cat_col}')
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # TAB 4: Correlations
        with tab4:
            st.subheader("🔗 Correlation Analysis")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(num_cols) >= 2:
                # Correlation matrix
                corr_matrix = df[num_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    aspect="auto",
                                    title="Correlation Heatmap",
                                    color_continuous_scale='RdBu_r',
                                    zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Show highest correlations
                st.subheader("Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(10)
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.warning("⚠️ Need at least 2 numerical columns for correlation analysis")
        
        # TAB 5: Filter Data
        with tab5:
            st.subheader("🔍 Filter Your Data")
            
            # Column selection
            filter_col = st.selectbox("Select column to filter", df.columns)
            
            if df[filter_col].dtype in [np.number, 'int64', 'float64']:
                # Numerical filtering
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                
                values = st.slider(f"Select range for {filter_col}", 
                                  min_val, max_val, (min_val, max_val))
                
                filtered_df = df[(df[filter_col] >= values[0]) & (df[filter_col] <= values[1])]
            else:
                # Categorical filtering
                unique_values = df[filter_col].unique().tolist()
                selected_values = st.multiselect(f"Select values for {filter_col}", 
                                                 unique_values, 
                                                 default=unique_values[:5] if len(unique_values) > 5 else unique_values)
                
                filtered_df = df[df[filter_col].isin(selected_values)]
            
            st.write(f"**Filtered Data: {len(filtered_df)} rows**")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please make sure you uploaded a valid CSV file.")

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    # Show example
    st.subheader("Example: What This App Can Do")
    st.markdown("""
    - **Data Preview**: View your data and column information
    - **Data Profile**: Get summary statistics, missing values, duplicates
    - **Visualizations**: Interactive histograms, scatter plots, box plots
    - **Correlations**: Analyze relationships between numerical variables
    - **Filter Data**: Filter and download subsets of your data
    """)
