import streamlit as st
import pandas as pd
import streamlit_shadcn_ui as ui



st.set_page_config(layout="wide", initial_sidebar_state="expanded",page_title="LinkdIn Job Dashboard " ,page_icon="📈")



intro_page=st.Page(page="pages/0_Intro.py",title="Intro",icon=":material/auto_awesome:")
main_page=st.Page(page="pages/Main.py",title="Main Page",icon=":material/home:")
analysis_page=st.Page(page="pages/1_Application_Analysis.py",title="Analysis",icon=":material/analytics:")
scrape_page=st.Page(page="pages/2_ScrapeAnd_Analyze_Skills.py",title="Description Analysis",icon=":material/thumb_up:")

nav=st.navigation({
            "Overview":[intro_page, main_page],
            "Insights":[analysis_page,scrape_page]
        })

nav.run()
