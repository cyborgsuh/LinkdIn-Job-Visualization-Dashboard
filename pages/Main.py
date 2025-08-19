import streamlit as st
import pandas as pd
import streamlit_shadcn_ui as ui

st.title('LinkedIn Job Application Dashboard')
st.write('Upload your job application data to get started.')
st.sidebar.warning('LinkedIn has multiple Job Application csv\'s, make sure to upload all of them')

st.sidebar.header('Upload Your Files')

uploaded_files = st.sidebar.file_uploader(
    "Upload your 'Job Applications.csv' file(s) from LinkedIn",
    accept_multiple_files=True,
    type=['csv'],
    key="job_apps_uploader"
)

# Optional: allow clearing persisted data
if 'df' in st.session_state:
    if st.sidebar.button('Clear uploaded data'):
        st.session_state.pop('df', None)
        st.session_state.pop('uploaded_filenames', None)
        st.rerun()

def render_overview(dataframe: pd.DataFrame) -> None:
    st.header('Data Overview')

    total_applications = len(dataframe)
    date_from = dataframe['Application Date'].min().date()
    date_to = dataframe['Application Date'].max().date()
    unique_companies = dataframe['Company Name'].nunique()

    cols = st.columns(4)
    with cols[0]:
        ui.metric_card(title="Total Applications", content=str(total_applications), description="Total jobs applied for")
    with cols[1]:
        ui.metric_card(title="Unique Companies", content=str(unique_companies), description="Number of unique companies")
    with cols[2]:
        ui.metric_card(title="Start Date", content=str(date_from), description="Oldest application")
    with cols[3]:
        ui.metric_card(title="End Date", content=str(date_to), description="Most recent application")

    st.dataframe(dataframe.head())

if uploaded_files:
    try:
        dfs = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True)
        df['Application Date'] = pd.to_datetime(df['Application Date'], errors='coerce', format='%m/%d/%y, %I:%M %p')
        df = df.dropna(subset=['Application Date'])
        st.session_state['df'] = df
        st.session_state['uploaded_filenames'] = [file.name for file in uploaded_files]
        st.success('Files uploaded successfully! Navigate to the analysis pages using the sidebar.')

        render_overview(df)

    except Exception as e:
        st.error(f'An error occurred: {e}')

elif 'df' in st.session_state:
    df = st.session_state['df']
    st.info('Using previously uploaded data stored in session.')
    if 'uploaded_filenames' in st.session_state and st.session_state['uploaded_filenames']:
        st.sidebar.caption('Previously uploaded files:')
        for name in st.session_state['uploaded_filenames']:
            st.sidebar.write(f"- {name}")
    render_overview(df)

else:
    st.info('Please upload your CSV files to begin.')

