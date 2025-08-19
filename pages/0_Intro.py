import streamlit as st


st.title("✨ LinkedIn Job Application Dashboard")
st.subheader("From raw exports to insights: applications, trends, and in‑demand skills")

with st.container(border=True):
	st.markdown("""
	### What this app does
	- Analyze your job application history: volumes, timing, top companies, and roles
	- Discover skill demand from job descriptions and track trends over time
	- Visualize how skills co‑occur together to reveal role expectations
	""")

col1, col2, col3 = st.columns(3)
with col1:
	with st.container(border=True):
		st.markdown("**📈 Applications**")
		st.caption("Daily/total trajectory, top companies, roles, time‑of‑day patterns")
with col2:
	with st.container(border=True):
		st.markdown("**🧠 Skills**")
		st.caption("Frequently requested skills + skill network from job descriptions")
with col3:
	with st.container(border=True):
		st.markdown("**📊 Trends**")
		st.caption("Trending skills over time to guide your learning focus")

st.markdown("---")

with st.container(border=True):
	st.markdown("""
	### How to export your data from LinkedIn
	1. Open LinkedIn and go to **Profile** → **Settings & Privacy**
	2. Navigate to **Data privacy** → **Get a copy of your data**
	3. Choose **Want something in particular?** and select:
	   - Job applications (CSV)
	   - Optionally other related items as needed
	4. Submit the request and wait for the email from LinkedIn
	5. Download the ZIP archive locally
	""")
	st.info("You will receive multiple CSVs for job applications. Upload all of them on the Main Page.")

with st.container(border=True):
	st.markdown("""
	### How to use this app
	- Go to **Main Page** → Upload all your `Job Applications.csv` files (you can select multiple)
	- Go to **Description Analysis** → Either:
	  - Click **Start Scraping** to fetch live job descriptions from your application links, or
	  - Upload your own scraped JSON/CSV
	- Explore the visualizations right on the **Description Analysis** page and the **Analysis** page
	""")
	st.caption("JSON accepted shapes: a list of flat job dicts, a list/dict keyed by URL → details. Required fields: title, company, description. Optional: location, link, skills.")

st.markdown("---")

st.markdown("<div style='text-align:center; font-size: 16px;'>Made with ♡ by suhaib</div>", unsafe_allow_html=True)
