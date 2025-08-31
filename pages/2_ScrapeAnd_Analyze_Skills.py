import streamlit as st
import pandas as pd
import requests
import bs4
import json
import re
from collections import Counter
import plotly.express as px
import time
from datetime import datetime, timedelta
import networkx as nx
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

st.title('Scrape and Analyze Job Skills')
st.markdown("---")

# Function to scrape job details from LinkedIn
@st.cache_data
def scrape_linkedin_job_details(job_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(job_url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None
        
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        
        # Extract job details with better error handling
        title = soup.find('h1', class_='top-card-layout__title')
        title = title.text.strip() if title else 'Unknown'
        
        company = soup.find('a', class_='topcard__org-name-link')
        company = company.text.strip() if company else 'Unknown'
        
        description = soup.find('div', class_='description__text')
        description = description.text.strip() if description else 'No description available'
        
        location = soup.find('span', class_='topcard__flavor--bullet')
        location = location.text.strip() if location else 'Unknown'
        
        return {
            'title': title, 
            'company': company, 
            'description': description, 
            'location': location,
            'scraped_at': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error scraping {job_url}: {str(e)}")
        return None

def extract_skills_from_description(text):
    """Extract skills from job description using comprehensive vocabulary"""
    skills_vocab = [
        # Programming Languages
        'python', 'sql', 'r', 'java', 'javascript', 'typescript', 'c++', 'c#', 'scala', 'go', 'rust',
        # Data Science & ML
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'nlp', 'computer vision',
        'neural networks', 'regression', 'classification', 'clustering', 'time series', 'reinforcement learning',
        # Cloud & Infrastructure
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab',
        # Data & Analytics
        'tableau', 'power bi', 'looker', 'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'snowflake',
        'bigquery', 'redshift', 'mongodb', 'postgresql', 'mysql', 'elasticsearch', 'redis',
        # Tools & Platforms
        'git', 'jira', 'confluence', 'slack', 'excel', 'powerpoint', 'word', 'outlook',
        # Specialized Skills
        'mlops', 'devops', 'data engineering', 'data visualization', 'big data', 'data mining',
        'statistics', 'analytics', 'business intelligence', 'etl', 'data modeling', 'data warehousing',
        'opencv', 'tesseract', 'bert', 'gpt', 'transformer', 'cnn', 'rnn', 'lstm', 'xgboost',
        'lightgbm', 'catboost', 'random forest', 'svm', 'k-means', 'hierarchical clustering',
        'anomaly detection', 'recommendation systems', 'a/b testing', 'hypothesis testing'
    ]
    
    found_skills = []
    text_lower = str(text).lower()
    
    for skill in skills_vocab:
        # Check for exact word boundaries
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            found_skills.append(skill)
        # Check for common abbreviations
        elif skill == 'machine learning' and re.search(r'\bml\b', text_lower):
            found_skills.append(skill)
        elif skill == 'artificial intelligence' and re.search(r'\bai\b', text_lower):
            found_skills.append(skill)
        elif skill == 'natural language processing' and re.search(r'\bnlp\b', text_lower):
            found_skills.append(skill)
        elif skill == 'computer vision' and re.search(r'\bcv\b', text_lower):
            found_skills.append(skill)
    
    return list(set(found_skills))  # Remove duplicates

def _format_seconds(total_seconds: float) -> str:
    try:
        seconds = int(total_seconds)
    except Exception:
        seconds = 0
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins} min {secs} sec"

def create_wordcloud_from_text(text, title="Word Cloud", max_words=100, width=800, height=400):
    """Create a word cloud from text and return as base64 encoded image"""
    try:
        # Clean and prepare text
        if not text or pd.isna(text):
            return None
        
        # Remove common stop words and clean text
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
        
        # Clean text - remove special characters and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        words = clean_text.split()
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if not filtered_words:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width, 
            height=height, 
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            collocations=False,
            random_state=42
        ).generate(' '.join(filtered_words))
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        
        # Convert to base64 for display in Streamlit
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def create_wordcloud_from_skills(skills_list, title="Skills Word Cloud", max_words=50, width=800, height=400):
    """Create a word cloud from skills list and return as base64 encoded image"""
    try:
        if not skills_list or not isinstance(skills_list, list):
            return None
        
        # Flatten skills if they're nested
        flat_skills = []
        for skill in skills_list:
            if isinstance(skill, str):
                flat_skills.append(skill)
            elif isinstance(skill, list):
                flat_skills.extend(skill)
        
        if not flat_skills:
            return None
        
        # Create frequency dictionary
        skill_freq = Counter(flat_skills)
        
        if not skill_freq:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width, 
            height=height, 
            background_color='white',
            max_words=max_words,
            colormap='plasma',
            collocations=False,
            random_state=42
        ).generate_from_frequencies(skill_freq)
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        
        # Convert to base64 for display in Streamlit
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    except Exception as e:
        st.error(f"Error creating skills word cloud: {str(e)}")
        return None

def _render_wordcloud_analysis(df_desc):
    """Render word cloud analysis section"""
    st.subheader("☁️ Word Cloud Analysis")
    
    # Create tabs for different word cloud types
    tab1, tab2, tab3 = st.tabs(["📝 Job Descriptions", "🛠️ Skills", "📊 Custom Analysis"])
    
    with tab1:
        st.markdown("### Job Descriptions Word Cloud")
        st.caption("Shows the most frequent words across all job descriptions")
        
        # Combine all descriptions
        all_descriptions = ' '.join([
            str(desc) for desc in df_desc['description'].dropna() 
            if desc and str(desc).strip() != 'No description available'
        ])
        
        if all_descriptions and len(all_descriptions.strip()) > 10:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Word cloud controls
                max_words_desc = st.slider("Max words for descriptions", 50, 200, 100, key="desc_words")
                width_desc = st.slider("Width", 400, 1200, 800, key="desc_width")
                height_desc = st.slider("Height", 300, 800, 400, key="desc_height")
            
            with col2:
                st.caption("💡 **Tip**: Adjust the sliders to customize your word cloud appearance")
            
            # Generate and display word cloud
            img_str = create_wordcloud_from_text(
                all_descriptions, 
                "Job Descriptions Word Cloud", 
                max_words_desc, 
                width_desc, 
                height_desc
            )
            
            if img_str:
                st.image(f"data:image/png;base64,{img_str}", use_container_width =True)
            else:
                st.info("No valid descriptions found for word cloud generation.")
        else:
            st.info("No job descriptions available for word cloud analysis.")
    
    with tab2:
        st.markdown("### Skills Word Cloud")
        st.caption("Shows the most frequent skills across all job postings")
        
        if 'skills' in df_desc.columns and not df_desc['skills'].empty:
            # Collect all skills
            all_skills = []
            for skills in df_desc['skills']:
                if isinstance(skills, list):
                    all_skills.extend(skills)
                elif isinstance(skills, str):
                    all_skills.extend([s.strip() for s in skills.split(',') if s.strip()])
            
            if all_skills:
                col1, col2 = st.columns([2, 1])
                with col1:
                    max_words_skills = st.slider("Max words for skills", 20, 100, 50, key="skills_words")
                    width_skills = st.slider("Width", 400, 1200, 800, key="skills_width")
                    height_skills = st.slider("Height", 300, 800, 400, key="skills_height")
                
                with col2:
                    st.caption("💡 **Tip**: Skills word clouds help identify the most in-demand technical skills")
                
                # Generate and display skills word cloud
                img_str = create_wordcloud_from_skills(
                    all_skills, 
                    "Skills Word Cloud", 
                    max_words_skills, 
                    width_skills, 
                    height_skills
                )
                
                if img_str:
                    st.image(f"data:image/png;base64,{img_str}", use_container_width =True)
                else:
                    st.info("Could not generate skills word cloud.")
            else:
                st.info("No skills found for word cloud analysis.")
        else:
            st.info("No skills data available for word cloud analysis.")
    
    with tab3:
        st.markdown("### Custom Text Analysis")
        st.caption("Analyze custom text or specific job descriptions")
        
        custom_text = st.text_area(
            "Enter custom text for analysis",
            height=150,
            placeholder="Paste job description text here for custom word cloud analysis..."
        )
        
        if custom_text and len(custom_text.strip()) > 10:
            col1, col2 = st.columns([2, 1])
            with col1:
                max_words_custom = st.slider("Max words", 50, 200, 100, key="custom_words")
                width_custom = st.slider("Width", 400, 1200, 800, key="custom_width")
                height_custom = st.slider("Height", 300, 800, 400, key="custom_height")
            
            with col2:
                st.caption("💡 **Tip**: Use this for analyzing specific job descriptions or custom text")
            
            if st.button("Generate Custom Word Cloud", type="primary"):
                img_str = create_wordcloud_from_text(
                    custom_text, 
                    "Custom Text Word Cloud", 
                    max_words_custom, 
                    width_custom, 
                    height_custom
                )
                
                if img_str:
                    st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
                else:
                    st.error("Could not generate word cloud from the provided text.")
        else:
            st.info("Enter some text above to generate a custom word cloud.")

def _render_frequently_requested_skills(df_desc: pd.DataFrame) -> None:
    if 'skills' not in df_desc.columns or df_desc['skills'].empty:
        st.info('No skills available to analyze.')
        return
    all_skills: list[str] = []
    for val in df_desc['skills']:
        if isinstance(val, list):
            all_skills.extend(val)
        elif isinstance(val, str):
            parts = re.split(r"[;,]", val)
            all_skills.extend([p.strip() for p in parts if p.strip()])
    if not all_skills:
        st.info('No skills available to analyze.')
        return
    top = pd.Series(all_skills).value_counts().head(20)
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation='h',
        title='Frequently Requested Skills',
        labels={'x': 'Count', 'y': 'Skill'},
        color=top.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# --- User-driven skill focus controls (applied to Trending + Network) ---
def _build_skills_focus(df_desc: pd.DataFrame, selected_skills: list[str], fallback_keywords: list[str]) -> pd.DataFrame:
    if df_desc.empty:
        return df_desc
    selected = {s.strip().lower() for s in selected_skills if str(s).strip()}
    fallbacks = {k.strip().lower() for k in fallback_keywords if str(k).strip()}
    if not selected and not fallbacks:
        return df_desc

    def compute_focus(row: pd.Series) -> list[str]:
        focus: set[str] = set()
        # From existing extracted skills
        if 'skills' in row and isinstance(row['skills'], list):
            for s in row['skills']:
                s_norm = str(s).strip().lower()
                if not selected or s_norm in selected:
                    focus.add(s)
        # From description via fallback keywords
        if fallbacks and 'description' in row and isinstance(row['description'], str):
            text = row['description']
            for kw in fallbacks:
                try:
                    if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
                        # Add the normalized keyword label if not already present
                        focus.add(kw)
                except Exception:
                    continue
        return sorted(focus)

    df_out = df_desc.copy()
    df_out['skills_focus'] = df_out.apply(compute_focus, axis=1)
    return df_out

# Modify renderers to prefer filtered skills if available
def _get_skills_column(df_desc: pd.DataFrame) -> str:
    return 'skills_focus' if 'skills_focus' in df_desc.columns else 'skills'

def _render_trending_skills_over_time(df_desc: pd.DataFrame) -> None:
    # Determine a date column to aggregate by (Year-Month)
    date_series = None
    if 'scraped_at' in df_desc.columns:
        date_series = pd.to_datetime(df_desc['scraped_at'], errors='coerce')
    elif 'date' in df_desc.columns:
        date_series = pd.to_datetime(df_desc['date'], errors='coerce')
    elif 'link' in df_desc.columns and 'df' in st.session_state and 'Job Url' in st.session_state['df'].columns:
        try:
            df_apps = st.session_state['df']
            merged = df_desc.merge(df_apps[['Job Url','Application Date']], left_on='link', right_on='Job Url', how='left')
            date_series = pd.to_datetime(merged['Application Date'], errors='coerce')
        except Exception:
            date_series = None

    if date_series is None or date_series.dropna().empty:
        st.info('No dates available to trend skills. Add a date field (scraped_at/date) or upload applications with Job Url to infer dates.')
        return

    df = df_desc.copy()
    df['_year_month'] = date_series.dt.to_period('M').astype(str)

    skills_col = _get_skills_column(df)

    # Expand rows to one record per skill
    records: list[dict] = []
    for ym, skills in zip(df['_year_month'], df[skills_col] if skills_col in df.columns else [[]]*len(df)):
        if isinstance(skills, list):
            for s in skills:
                records.append({'YearMonth': ym, 'Skill': s})
        elif isinstance(skills, str):
            parts = re.split(r"[;,]", skills)
            for s in [p.strip() for p in parts if p.strip()]:
                records.append({'YearMonth': ym, 'Skill': s})

    if not records:
        st.info('No skill records available to trend.')
        return

    trend_df = pd.DataFrame(records)
    top_skills = trend_df['Skill'].value_counts().head(8).index
    trend_top = (
        trend_df[trend_df['Skill'].isin(top_skills)]
        .groupby(['YearMonth','Skill']).size().reset_index(name='Count')
        .sort_values('YearMonth')
    )

    fig = px.line(trend_top, x='YearMonth', y='Count', color='Skill', title='Trending Skills Over Time')
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)

def _render_skill_combination_network(df_desc: pd.DataFrame) -> None:
    skills_col = _get_skills_column(df_desc)
    if skills_col not in df_desc.columns:
        st.info('No skills to build a network.')
        return
    from itertools import combinations
    pair_counts: dict[tuple[str, str], int] = {}
    for val in df_desc[skills_col]:
        skills = val if isinstance(val, list) else []
        uniq = sorted(set([str(s) for s in skills]))
        for a, b in combinations(uniq, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1
    if not pair_counts:
        st.info('No skill combinations found.')
        return
    edges_sorted = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:100]
    G = nx.Graph()
    for (a, b), w in edges_sorted:
        G.add_edge(a, b, weight=w)
    if len(G.nodes()) < 2:
        st.info('Not enough connections to display network.')
        return
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#94a3b8'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        deg = G.degree[n]
        node_size.append(10 + deg * 3)
        node_color.append(deg)
        node_text.append(f"{n} (deg {deg})")
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=[n for n in G.nodes()], textposition='top center',
        marker=dict(showscale=True, colorscale='Viridis', color=node_color, size=node_size, line=dict(width=2, color='white'),
                    colorbar=dict(title='Node Degree', thickness=12, len=0.5)),
        hovertext=node_text, hoverinfo='text'
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='Skill Combination Network', showlegend=False, margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)

def _render_overview_metrics(df_desc: pd.DataFrame) -> None:
    total_jobs = len(df_desc)
    # Choose skills column
    skills_col = 'skills_focus' if 'skills_focus' in df_desc.columns else ('skills' if 'skills' in df_desc.columns else None)
    distinct_skills = 0
    avg_skills_per_job = 0.0
    if skills_col:
        all_skills: list[str] = []
        lengths: list[int] = []
        for val in df_desc[skills_col]:
            if isinstance(val, list):
                all_skills.extend(val)
                lengths.append(len(val))
            elif isinstance(val, str):
                parts = re.split(r"[;,]", val)
                cleaned = [p.strip() for p in parts if p.strip()]
                all_skills.extend(cleaned)
                lengths.append(len(cleaned))
        if all_skills:
            distinct_skills = len(set(map(str, all_skills)))
        if lengths:
            avg_skills_per_job = sum(lengths) / len(lengths)
    # Extract experience in years from description
    exp_series = pd.to_numeric(
        df_desc.get('description', pd.Series(dtype=str)).astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", expand=False), errors='coerce'
    )
    avg_exp = float(exp_series.mean()) if not exp_series.dropna().empty else 0.0
    exp_coverage = int(exp_series.notna().sum())

    c2, c3, c4 = st.columns(3)

    with c2:
        st.metric("Distinct Skills Found", f"{distinct_skills}")
    with c3:
        st.metric("Avg Skills per Job", f"{avg_skills_per_job:.1f}")
    with c4:
        st.metric("Avg Experience (yrs)", f"{avg_exp:.1f}", help=f"Extracted from descriptions; coverage: {exp_coverage} rows")

def _df_from_uploaded_json(raw_json) -> pd.DataFrame:
    """Accepts either:
    - List[Dict[str, Any]] where each dict is a flat job object
    - List[Dict[str, Dict]] where each dict has a single key (URL) mapping to details
    - Dict[str, Dict] mapping URL->details
    Returns a DataFrame with columns including 'title','company','description', optional 'location','skills','link'
    """
    try:
        if isinstance(raw_json, list):
            rows = []
            nested_detected = False
            for item in raw_json:
                if isinstance(item, dict):
                    if len(item) == 1:
                        # Possibly { "http://...": { title, company, ... } }
                        (link, details) = next(iter(item.items()))
                        if isinstance(details, dict):
                            nested_detected = True
                            row = details.copy()
                            row['link'] = link
                            rows.append(row)
                    else:
                        rows.append(item)
            if nested_detected:
                return pd.DataFrame(rows)
            return pd.DataFrame(rows)
        # Case 2: dict mapping link->details
        if isinstance(raw_json, dict):
            rows = []
            for link, details in raw_json.items():
                if isinstance(details, dict):
                    row = details.copy()
                    row['link'] = link
                    rows.append(row)
            if rows:
                return pd.DataFrame(rows)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()




# --- Embeddings + Clustering (LM Studio) ---

def _render_lmstudio_instructions() -> None:
    with st.expander("How to run LM Studio for embeddings", expanded=False):
        st.markdown("""
        1) Open LM Studio → start the local server (Server icon → Start Server)
        2) Copy the Base URL (default: `http://localhost:1234/v1`)
        3) Choose an embeddings-capable model (e.g., `text-embedding-nomic-embed-text`)
        4) Any API key works for LM Studio (we don't send it externally)
        """)


def _embed_and_cluster_lmstudio(df_desc: pd.DataFrame, text_col: str, base_url: str, model: str, n_clusters: int, projection: str = "2D") -> pd.DataFrame | None:
    try:
        from openai import OpenAI 
    except Exception:
        st.error("Missing dependency: install with `pip install openai`. Then rerun.")
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
    except Exception:
        st.error("Missing dependency: install with `pip install scikit-learn`. Then rerun.")
        return None

    # Prepare texts (dedent/strip)
    texts = df_desc[text_col].astype(str).fillna("").map(lambda t: t.replace("\n", " ").strip()).tolist()
    if not texts:
        st.info("No descriptions available to embed.")
        return None

    client = OpenAI(base_url=base_url, api_key="lm-studio")

    vectors: list[list[float]] = []
    total = len(texts)
    batch_size = 32 if total > 64 else 16
    completed = 0

    st.caption("Embedding descriptions via LM Studio…")
    progress_placeholder = st.empty()
    st_progress = None
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        except Exception as e:
            st.error(f"Embedding request failed on batch {i//batch_size+1}: {e}")
            return None
        completed += len(batch)
        percent = int(completed/total*100) if total else 100
        try:
            with progress_placeholder:
                ui.progress_bar(value=percent, max=100, label=f"Embedding {completed}/{total}", key="emb_prog")
        except Exception:
            if st_progress is None:
                st_progress = st.progress(0.0)
            st_progress.progress(completed/total)

    if not vectors:
        st.error("Received no embeddings. Check LM Studio server/model.")
        return None

    # KMeans
    try:
        with st.spinner("Clustering with KMeans…"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(vectors)
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        return None

    # Projection
    try:
        if projection == "3D":
            with st.spinner("Projecting to 3D with PCA…"):
                pca = PCA(n_components=3, random_state=42)
                pts = pca.fit_transform(vectors)
            vis_df = pd.DataFrame({
                "x": [p[0] for p in pts],
                "y": [p[1] for p in pts],
                "z": [p[2] for p in pts],
                "cluster": labels,
                "title": df_desc.get("title", pd.Series([""]*len(texts))).astype(str).tolist(),
                "company": df_desc.get("company", pd.Series([""]*len(texts))).astype(str).tolist(),
            })
        else:
            with st.spinner("Projecting to 2D with PCA…"):
                pca = PCA(n_components=2, random_state=42)
                pts = pca.fit_transform(vectors)
            vis_df = pd.DataFrame({
                "x": [p[0] for p in pts],
                "y": [p[1] for p in pts],
                "cluster": labels,
                "title": df_desc.get("title", pd.Series([""]*len(texts))).astype(str).tolist(),
                "company": df_desc.get("company", pd.Series([""]*len(texts))).astype(str).tolist(),
            })
    except Exception as e:
        st.error(f"Projection failed: {e}")
        return None

    return vis_df


def _render_embeddings_vis(vis_df: pd.DataFrame, projection: str, n_clusters: int, df_desc: pd.DataFrame, text_col: str) -> None:
    if projection == "3D" and {"x","y","z","cluster"}.issubset(vis_df.columns):
        fig = px.scatter_3d(
            vis_df, x="x", y="y", z="z", color="cluster",
            hover_data=[c for c in ["title","company","location"] if c in vis_df.columns],
            title=f"Job Descriptions Clusters (3D, k={n_clusters})"
        )
    else:
        fig = px.scatter(
            vis_df, x="x", y="y", color="cluster",
            hover_data=[c for c in ["title","company","location"] if c in vis_df.columns],
            title=f"Job Descriptions Clusters (2D, k={n_clusters})"
        )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sample of clustered descriptions (first 20 rows)")
    st.dataframe(vis_df.assign(description=df_desc[text_col]).head(20), use_container_width=True)


def _lm_sig(df_desc: pd.DataFrame, text_col: str, base_url: str, model: str, n_clusters: int, projection: str) -> str:
    import hashlib, json
    texts = df_desc[text_col].astype(str).fillna("").tolist()
    payload = json.dumps({"t": texts, "base": base_url, "m": model, "k": n_clusters, "p": projection}, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


if 'df' not in st.session_state:
    st.warning('⚠️ Please upload your data on the main page first.')
    st.info('This page requires job application data with LinkedIn URLs to scrape job descriptions.')
else:
    df_apps = st.session_state['df']
    
    with st.container(border=True):
        st.subheader("📋 Scraping Overview")
        
        total_applications = len(df_apps)
        has_urls = 'Job Url' in df_apps.columns
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Applications", total_applications)
        with col2:
            st.metric("URLs Available", "✅ Yes" if has_urls else "❌ No")
        with col3:
            if has_urls:
                valid_urls = df_apps['Job Url'].notna().sum()
                st.metric("Valid URLs", valid_urls)
        
        if not has_urls:
            st.error("❌ No 'Job Url' column found in your data. Cannot proceed with scraping.")
            st.info("Make sure your CSV contains a column named 'Job Url' with LinkedIn job posting URLs.")
        else:
            has_desc = 'df_desc' in st.session_state and not st.session_state['df_desc'].empty if 'df_desc' in st.session_state else False
            if has_desc:
                st.success('Scraped data detected. Hiding start/upload controls and showing analysis below.')
            else:
                with st.container(border=True):
                    st.subheader("📤 Upload Your Own Scraped Data (Optional)")
                    with st.expander("What format should my CSV/JSON have?", expanded=False):
                        st.markdown("- **Required fields**: `title`, `company`, `description`")
                        st.markdown("- **Optional fields**: `location`, `link`, `skills` (list or comma-separated)")
                        st.markdown("- **Notes**: If `skills` is missing, they will be extracted from `description` automatically.")
                        st.markdown("Accepted JSON formats:")
                        st.code('[\n  {\n    "title": "Data Scientist",\n    "company": "Acme Corp",\n    "description": "We need Python, SQL and ML.",\n    "location": "Dubai",\n    "link": "https://www.linkedin.com/jobs/view/123",\n    "skills": ["python", "sql", "machine learning"]\n  }\n]')
                        st.markdown("Or nested by URL:")
                        st.code('[\n  {\n    "http://www.linkedin.com/jobs/view/3908193290": {\n      "title": "Data Scientist",\n      "company": "Acme Corp",\n      "description": "We need Python, SQL and ML.",\n      "location": "Dubai",\n      "skills": ["python", "sql", "machine learning"]\n    }\n  }\n]')
                        st.markdown("Example CSV:")
                        sample_csv = "link,title,company,description,location\nhttp://www.linkedin.com/jobs/view/3908193290,Data Engineer,Lumi | لومي,We need someone who can work with data,Dubai AE\n"
                        st.code(sample_csv)
                        colt1, colt2 = st.columns(2)
                        with colt1:
                            st.download_button("Download CSV template", data=sample_csv, file_name="scraped_template.csv", mime="text/csv")
                        with colt2:
                            sample_json = '[\n  {\n    "title": "Data Scientist",\n    "company": "Acme Corp",\n    "description": "We need Python, SQL and ML.",\n    "location": "Dubai",\n    "link": "https://www.linkedin.com/jobs/view/123",\n    "skills": ["python", "sql", "machine learning"]\n  }\n]'
                            st.download_button("Download JSON template", data=sample_json, file_name="scraped_template.json", mime="application/json")
                    st.caption("Provide a JSON or CSV with the fields described above.")
                    uploaded_data = st.file_uploader("Upload JSON or CSV", type=["json", "csv"], key="upload_scraped")
                    if uploaded_data is not None:
                        try:
                            if uploaded_data.name.lower().endswith('.json'):
                                data = json.load(uploaded_data)
                                df_desc_custom = _df_from_uploaded_json(data)
                            else:
                                df_desc_custom = pd.read_csv(uploaded_data)
                            # Validate required columns
                            required_cols = {"title", "company", "description"}
                            df_desc_custom.columns = [c.lower() for c in df_desc_custom.columns]
                            missing = required_cols - set(df_desc_custom.columns)
                            if missing:
                                st.error(f"Missing required fields: {', '.join(sorted(missing))}")
                            else:
                                if 'skills' in df_desc_custom.columns:
                                    def _normalize_skills(val):
                                        if isinstance(val, list):
                                            return val
                                        try:
                                            import ast
                                            parsed = ast.literal_eval(str(val))
                                            if isinstance(parsed, list):
                                                return parsed
                                        except Exception:
                                            pass
                                        if isinstance(val, str):
                                            parts = re.split(r"[;,]", val)
                                            return [s.strip() for s in parts if s.strip()]
                                        return []
                                    df_desc_custom['skills'] = df_desc_custom['skills'].apply(_normalize_skills)
                                else:
                                    df_desc_custom['skills'] = df_desc_custom['description'].apply(extract_skills_from_description)
                                for opt in ['location','link']:
                                    if opt not in df_desc_custom.columns:
                                        df_desc_custom[opt] = None
                                st.session_state['df_desc'] = df_desc_custom
                                st.success("Custom scraped data loaded. Analysis will appear below.")
                        except Exception as e:
                            st.error(f"Failed to parse uploaded file: {e}")

                st.markdown("---")
                st.subheader("🚀 Start Scraping")
                start_clicked = st.button('▶️ Start Scraping Job Descriptions', key='scrape_button', type="primary")

                if 'cancel_scrape' not in st.session_state:
                    st.session_state['cancel_scrape'] = False

                if start_clicked:
                    st.info("💡 You can keep this running in the background and continue your work.")

                    cancel_clicked = st.button("✖️ Cancel Scraping", key="btn_cancel")
                    if cancel_clicked:
                        st.session_state['cancel_scrape'] = True

                    progress_container = st.container()
                    status_container = st.container()
                    results_container = st.container()

                    with progress_container:
                        st.subheader("📈 Scraping Progress")
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        time_estimate = st.empty()

                    with status_container:
                        status_text = st.empty()
                        current_job = st.empty()

                    start_time = time.time()
                    all_jobs = []
                    successful_scrapes = 0
                    failed_scrapes = 0

                    df_batch = df_apps
                    total_count = len(df_batch)

                    for i, (idx, row) in enumerate(df_batch.iterrows()):
                        if st.session_state.get('cancel_scrape'):
                            status_text.warning("Scraping cancelled by user.")
                            break

                        # Update progress
                        progress = (i + 1) / total_count
                        progress_bar.progress(progress)
                        progress_text.text(f"Progress: {i + 1}/{total_count} ({progress*100:.1f}%)")

                        # Update current job
                        current_job.text(f"🔄 Scraping: {row.get('Job Title', 'Unknown Title')} at {row.get('Company Name', 'Unknown Company')}")

                        # Update time estimate (mins/secs)
                        elapsed_time = time.time() - start_time
                        if i > 0:
                            avg_time_per_job = elapsed_time / i
                            remaining_jobs = total_count - i - 1
                            estimated_remaining = avg_time_per_job * remaining_jobs
                            time_estimate.text(f"⏱️ Estimated time remaining: {_format_seconds(estimated_remaining)}")

                        job_url = row['Job Url']
                        if pd.isna(job_url):
                            failed_scrapes += 1
                            continue

                        job_details = scrape_linkedin_job_details(job_url)

                        if job_details:
                            job_details['link'] = job_url
                            job_details['original_index'] = idx
                            job_details['skills'] = extract_skills_from_description(job_details['description'])
                            all_jobs.append(job_details)
                            successful_scrapes += 1
                        else:
                            failed_scrapes += 1

                        time.sleep(1)

                    if not st.session_state.get('cancel_scrape'):
                        progress_bar.progress(1.0)
                        progress_text.text(f"✅ Complete! {successful_scrapes}/{total_count} jobs scraped successfully")
                        time_estimate.text(f"⏱️ Total time: {_format_seconds(time.time() - start_time)}")

                    # Build df_desc and store
                    if successful_scrapes > 0:
                        df_desc = pd.DataFrame(all_jobs)
                        st.session_state['df_desc'] = df_desc

                    with results_container:
                        st.markdown("---")
                        st.subheader("📊 Scraping Results")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.metric("✅ Successful", successful_scrapes)
                        with col_r2:
                            st.metric("❌ Failed", failed_scrapes)
                        with col_r3:
                            rate = (successful_scrapes/total_count)*100 if total_count else 0
                            st.metric("📊 Success Rate", f"{rate:.1f}%")

if 'df_desc' in st.session_state and not st.session_state['df_desc'].empty:
    st.markdown("---")
    st.subheader("🔍 Skills Analysis from Scraped Data")
    df_desc = st.session_state['df_desc']

    with st.container(border=True):
        _render_overview_metrics(df_desc)

    with st.container(border=True):
        _render_frequently_requested_skills(df_desc)

    with st.container(border=True):
        st.markdown("### Focus the analysis (applies to the charts below)")
        st.caption("These filters do not affect the 'Frequently Requested Skills' chart above.")
        selected_skills_input = st.text_input(
            "Skills to include (comma-separated)",
            help="Only these skills will be analyzed in the charts below. Leave empty to use all extracted skills."
        )
        selected_skills = [s.strip() for s in selected_skills_input.split(',')] if selected_skills_input else []
        filtered_df_desc = _build_skills_focus(df_desc, selected_skills, []) if selected_skills else df_desc

    with st.container(border=True):
        st.subheader("Trending Skills Over Time (filtered)")
        _render_trending_skills_over_time(filtered_df_desc)
    with st.container(border=True):
        st.subheader("Skill Combination Network (filtered)")
        _render_skill_combination_network(filtered_df_desc)

    with st.container(border=True):
        _render_wordcloud_analysis(df_desc)

    with st.container(border=True):
        st.subheader("🔬 Embed and Cluster Job Descriptions (LM Studio)")
        _render_lmstudio_instructions()
        colc1, colc2, colc3, colc4 = st.columns([2,2,1,1])
        with colc1:
            text_col = st.selectbox(
                "Description column",
                options=[c for c in st.session_state['df_desc'].columns if c.lower() in ["description", "desc", "text"]] or list(st.session_state['df_desc'].columns),
                index=0
            )
            base_url = st.text_input("LM Studio Base URL", value="http://localhost:1234/v1")
        with colc2:
            model = st.text_input("Embeddings Model", value="text-embedding-nomic-embed-text")
        with colc3:
            n_clusters = st.number_input("Clusters", min_value=2, max_value=12, value=5, step=1)
        with colc4:
            projection = st.selectbox("Projection", options=["2D", "3D"], index=0)
        sig = _lm_sig(st.session_state['df_desc'], text_col, base_url, model, int(n_clusters), projection)
        cached = st.session_state.get('lmstudio_cache')

        run_clicked = st.button("Compute embeddings and cluster", type="primary")

        if run_clicked:
            vis = _embed_and_cluster_lmstudio(st.session_state['df_desc'], text_col, base_url, model, int(n_clusters), projection)
            if vis is not None:
                st.session_state['lmstudio_cache'] = {
                    'sig': sig,
                    'vis_df': vis,
                    'projection': projection,
                    'n_clusters': int(n_clusters),
                    'text_col': text_col,
                }
                _render_embeddings_vis(vis, projection, int(n_clusters), st.session_state['df_desc'], text_col)
        elif cached and cached.get('sig') == sig:
            _render_embeddings_vis(cached['vis_df'], cached['projection'], cached['n_clusters'], st.session_state['df_desc'], cached['text_col'])
