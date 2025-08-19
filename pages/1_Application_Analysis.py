import re
from typing import Optional, List, Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set page configuration

st.title('Application Analysis')

# Theming for Plotly charts
px.defaults.template = "plotly_white"
THEME_COLORS = [
    "#0ea5e9",  # sky-500
    "#22c55e",  # green-500
    "#ef4444",  # red-500
    "#f59e0b",  # amber-500
    "#a855f7",  # violet-500
    "#14b8a6",  # teal-500
    "#e11d48",  # rose-600
    "#84cc16",  # lime-500
    "#06b6d4",  # cyan-500
    "#f472b6",  # pink-400
]

@st.cache_data
def find_first_column(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    return next((col for col in candidates if col in dataframe.columns), None)

@st.cache_data
def process_time_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Process time-series data for applications."""
    try:
        return (dataframe
                .assign(**{'Application Date': pd.to_datetime(dataframe['Application Date'], errors='coerce')})
                .set_index('Application Date')
                .resample('D')
                .size()
                .reset_index(name='Applications'))
    except Exception as e:
        st.error(f"Error processing time data: {str(e)}")
        return pd.DataFrame(columns=['Application Date', 'Applications'])

def create_base_figure(title: str) -> Dict[str, Any]:
    """Create base figure layout settings."""
    return {
        'margin': dict(l=8, r=8, t=48, b=8),
        'title': title,
        'template': "plotly_white"
    }

def render_over_time_line(dataframe: pd.DataFrame, mode: str = 'total') -> None:
    """Render time series plot of applications."""
    try:
        daily = process_time_data(dataframe)
        if daily.empty:
            st.warning("No valid time series data available.")
            return

        df_plot = daily.copy()
        if mode != 'daily':
            df_plot['Applications'] = df_plot['Applications'].cumsum()
            title = 'Number of Applications Over Time'
        else:
            title = 'Daily Application Rate'

        fig = px.line(
            df_plot,
            x='Application Date',
            y='Applications',
            title=title,
            color_discrete_sequence=[THEME_COLORS[0]],
        )
        fig.update_layout(**create_base_figure(title))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering time series plot: {str(e)}")

def render_top_categories_bar(series: pd.Series, title: str, x_label: str, y_label: str, color_index: int = 1) -> None:
    top_items = series.value_counts().nlargest(15).reset_index()
    top_items.columns = [y_label, x_label]
    fig = px.bar(
        top_items,
        x=x_label,
        y=y_label,
        orientation='h',
        title=title,
        color_discrete_sequence=[THEME_COLORS[color_index % len(THEME_COLORS)]],
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def render_pie(series: pd.Series, title: str, color_index: int = 2) -> None:
    counts = series.value_counts()
    if counts.empty:
        st.info('No data available for this chart.')
        return
    fig = px.pie(values=counts.values, names=counts.index, title=title, color_discrete_sequence=THEME_COLORS)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)
def extract_keywords_from_titles(titles: pd.Series, top_n: int = 15) -> pd.Series:
    if titles.empty:
        return pd.Series(dtype=int)
    stopwords = set([
        'the','a','an','and','or','to','of','in','for','on','with','by','at','from','as','is','are','be','senior','jr','sr','ii','iii','iv','remote','hybrid','lead','principal','manager','specialist','associate'
    ])
    word_counts: dict[str, int] = {}
    for title in titles.dropna().astype(str):
        for token in title.replace('/',' ').replace('-',' ').split():
            token_clean = ''.join([ch for ch in token.lower() if ch.isalpha()])
            if len(token_clean) <= 2 or token_clean in stopwords:
                continue
            word_counts[token_clean] = word_counts.get(token_clean, 0) + 1
    if not word_counts:
        return pd.Series(dtype=int)
    return pd.Series(word_counts).sort_values(ascending=False).head(top_n)



@st.cache_data()
def render_keywords_bar(titles: pd.Series) -> None:
    keywords = extract_keywords_from_titles(titles)
    if keywords.empty:
        st.info('No keywords could be extracted from job titles.')
        return
    fig = px.bar(
        x=keywords.values,
        y=keywords.index,
        orientation='h',
        title='Most Common Keywords in Job Titles',
        labels={'x': 'Frequency', 'y': 'Keyword'},
        color_discrete_sequence=[THEME_COLORS[3]],
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def render_day_hour_distributions(dataframe: pd.DataFrame) -> None:
    date_series = pd.to_datetime(dataframe['Application Date'], errors='coerce')
    day_of_week = date_series.dt.day_name()
    hour_of_day = date_series.dt.hour

    day_counts = day_of_week.value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    hour_counts = hour_of_day.value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_day = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            title='Applications by Day of Week',
            labels={'x': 'Day', 'y': 'Applications'},
            color_discrete_sequence=[THEME_COLORS[4]],
        )
        fig_day.update_layout(margin=dict(l=8, r=8, t=48, b=8))
        st.plotly_chart(fig_day, use_container_width=True)
    with col2:
        fig_hour = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            title='Applications by Hour of Day',
            labels={'x': 'Hour (24h)', 'y': 'Applications'},
            color_discrete_sequence=[THEME_COLORS[5]],
        )
        fig_hour.update_layout(margin=dict(l=8, r=8, t=48, b=8))
        st.plotly_chart(fig_hour, use_container_width=True)

def render_monthly_bar(dataframe: pd.DataFrame) -> None:
    dates = pd.to_datetime(dataframe['Application Date'], errors='coerce')
    month_names = dates.dt.month_name()
    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    counts = month_names.value_counts().reindex(month_order)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title='Applications by Month',
        labels={'x': 'Month', 'y': 'Applications'},
        color_discrete_sequence=[THEME_COLORS[2]],
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def process_salary_data(dataframe: pd.DataFrame, salary_column: str, title_column: Optional[str]) -> Dict[str, Any]:
    """Process salary data for visualization."""
    try:
        salary_series = pd.to_numeric(dataframe[salary_column], errors='coerce')
        salary_df = dataframe.copy()
        salary_df['_salary_numeric'] = salary_series
        salary_df = salary_df.dropna(subset=['_salary_numeric'])
        
        result = {'salary_df': salary_df, 'title_agg': None}
        
        if not salary_df.empty and title_column:
            agg = (salary_df.groupby(title_column)['_salary_numeric']
                  .agg(['mean', 'count'])
                  .query('count >= 2')
                  .sort_values('mean', ascending=True)
                  .tail(20)
                  .rename(columns={'mean': 'Average Salary'}))
            result['title_agg'] = agg if not agg.empty else None
            
        return result
    except Exception as e:
        st.error(f"Error processing salary data: {str(e)}")
        return {'salary_df': pd.DataFrame(), 'title_agg': None}

def render_salary_charts(dataframe: pd.DataFrame, salary_column: str, title_column: Optional[str]) -> None:
    """Render salary distribution and average salary by title charts."""
    data = process_salary_data(dataframe, salary_column, title_column)
    salary_df = data['salary_df']
    
    if salary_df.empty:
        st.info('No numeric salary data available.')
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(
            salary_df,
            y='_salary_numeric',
            title='Distribution of Salary Expectations',
            labels={'_salary_numeric': 'Salary Expectation'},
            color_discrete_sequence=[THEME_COLORS[6]]
        )
        fig_box.update_layout(**create_base_figure('Distribution of Salary Expectations'))
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col2:
        if title_column and data['title_agg'] is not None:
            fig_bar = px.bar(
                data['title_agg'],
                x='Average Salary',
                y=data['title_agg'].index,
                orientation='h',
                title='Average Salary Expectations by Job Title (min 2)',
                color_discrete_sequence=[THEME_COLORS[7]]
            )
            fig_bar.update_layout(
                **create_base_figure('Average Salary Expectations by Job Title'),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info('Not enough salary data by title available (need at least 2 entries per title).')

def render_application_questions(dataframe: pd.DataFrame, questions_column: str) -> None:
    questions_series = dataframe[questions_column].dropna().astype(str)
    if questions_series.empty:
        st.info('No application questions found.')
        return
    all_questions: List[str] = []
    for text in questions_series:
        # Split by common delimiters
        parts = [p.strip() for p in re.split(r'[\n\r;|]+', text) if p.strip()]
        all_questions.extend(parts)
    if not all_questions:
        st.info('No application questions found.')
        return
    counts = pd.Series(all_questions).value_counts().head(15)
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        title='Most Common Application Questions',
        labels={'x': 'Frequency', 'y': 'Question'},
        color_discrete_sequence=[THEME_COLORS[8]],
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate and prepare the dataframe for analysis."""
    required_columns = {
        'company': ['Company Name', 'Company'],
        'title': ['Job Title', 'Title'],
        'resume': ['Resume Name', 'Resume', 'Resume Version'],
        'salary': ['Salary_Expectation', 'Salary Expectation'],
        'questions': ['Application Questions', 'Questions', 'Screening Questions']
    }
    
    result = {}
    for key, candidates in required_columns.items():
        result[key] = find_first_column(df, candidates)
    
    # Validate date column
    if 'Application Date' in df.columns:
        try:
            pd.to_datetime(df['Application Date'], errors='raise')
            result['date_valid'] = True
        except:
            result['date_valid'] = False
    else:
        result['date_valid'] = False
    
    return result

if 'df' not in st.session_state:
    st.warning('Please upload your data on the main page first.')
else:
    df = st.session_state['df']
    
    # Validate and get column mappings
    validation = validate_dataframe(df)
    
    if not validation['date_valid']:
        st.error("The 'Application Date' column is missing or contains invalid dates.")
        st.stop()
    
    # Get validated column names
    company_col = validation['company']
    title_col = validation['title']
    resume_col = validation['resume']
    salary_col = validation['salary']
    questions_col = validation['questions']

    # 1) Applications over time (full width)
    with st.container(key="card_over_time", border=True):
        st.subheader('Applications Over Time')
        view_mode = st.radio('View', options=['Total over time', 'Daily rate'], horizontal=True, key='apps_time_view')
        if view_mode == 'Daily rate':
            render_over_time_line(df, mode='daily')
        else:
            render_over_time_line(df, mode='total')

    # 2) Top companies and job titles
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(key="card_top_companies", border=True):
            st.subheader('Top Companies by Applications')
            if company_col:
                render_top_categories_bar(df[company_col], 'Top Companies Applied To', 'Number of Applications', 'Company', color_index=1)
            else:
                st.info('Company column not found.')
    with col_b:
        with st.container(key="card_top_titles", border=True):
            st.subheader('Most Common Job Titles')
            if title_col:
                render_top_categories_bar(df[title_col], 'Most Common Job Titles', 'Number of Applications', 'Job Title', color_index=2)
            else:
                st.info('Job Title column not found.')

    # 3) Keywords in job titles
    with st.container(key="card_keywords", border=True):
        st.subheader('Title Keywords')
        if title_col:
            render_keywords_bar(df[title_col])
        else:
            st.info('Job Title column not found.')

    # 4) Resume versions and applications by month
    col_e, col_f = st.columns(2)
    with col_e:
        with st.container(key="card_resume", border=True):
            st.subheader('Resume Versions Used')
            if resume_col:
                render_pie(df[resume_col], 'Resume Versions Used in Applications', color_index=4)
            else:
                st.info('Resume column not found.')
    with col_f:
        with st.container(key="card_monthly", border=True):
            st.subheader('Applications by Month')
            render_monthly_bar(df)

    # 5) Day and hour distributions
    with st.container(key="card_day_hour", border=True):
        st.subheader('When Do You Apply Most?')
        render_day_hour_distributions(df)

    # 6) Salary expectations (optional)
    if salary_col:
        with st.container(key="card_salary", border=True):
            st.subheader('Salary Expectations')
            render_salary_charts(df, salary_col, title_col)
