
import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.colors import to_hex

st.set_page_config(page_title="Reddit Data Generator", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Creation", "Proposed Topics", "Sentiment Analysis", "Visualization"])

if page == "Dataset Creation":
    st.title("Reddit Dataset Generator (Demo Mode)")
    st.info("⚠ This is a static demo. Data fetching has been disabled.")
    st.text_input("Subreddit Name", "worldnews", disabled=True)
    st.text_area("Enter search queries (one per line)", disabled=True)
    st.text_input("Enter filename to save JSON", "reddit_data.json", disabled=True)
    st.button("Fetch Data", disabled=True)

elif page == "Proposed Topics":
    st.title("Topic Modeling with LDA (Demo Mode)")
    st.info("⚠ This is a static demo. Upload disabled. Using preprocessed files.")
    dataset_choice = st.selectbox("Choose a dataset", ["reddit", "twitter"])
    num_topics = st.number_input("Enter number of topics (4–7):", min_value=4, max_value=7, value=5, step=1)
    filename = f"data/proposed_topics_{dataset_choice}_{num_topics}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        st.subheader(f"Top {num_topics} Topics from {dataset_choice.capitalize()}")
        for topic in df["Topic"]:
            st.write(topic)
    else:
        st.warning(f"No static file found at `{filename}`.")
    
    html_path = f"data/lda_vis_{dataset_choice}_{num_topics}.html"
    if os.path.exists(html_path):
        st.subheader("LDA Topic Visualization")
        with open(html_path, "r", encoding="utf-8") as f:
            html_string = f.read()
        st.components.v1.html(html_string, width=1700, height=1200)
    else:
        st.warning(f"No LDA visualization file found at {html_path}")

elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis and Topic Assignment (Demo Mode)")
    st.info("⚠ This is a static demo. File upload and topic input are disabled.")
    dataset_choice = st.selectbox("Choose a dataset", ["reddit", "twitter"])
    filename = f"data/sentiment_{dataset_choice}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        st.session_state.topic_emotions_df = df
        rows_per_page = 10
        total_pages = (len(df) // rows_per_page) + (1 if len(df) % rows_per_page else 0)
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page_num - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        df_paginated = df.iloc[start_idx:end_idx]
        st.subheader("Post Topics")
        edited_df = st.data_editor(
            df_paginated[["Select", "Title", "Assigned Topic", "Assigned Emotion"]],
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select a row"),
                "Title": st.column_config.TextColumn("Title"),
            }
        )
        selected_rows = df[df["Title"].isin(edited_df[edited_df["Select"]]["Title"])]
        if not selected_rows.empty:
            st.markdown("---")
            for _, row in selected_rows.iterrows():
                st.write(f"### {row['Title']}")
                st.write(f"**Post Content:** {row['Post']}")
                st.write(f"**Assigned Topic:** {row['Assigned Topic']}")
                st.write(f"**Assigned Emotion:** {row['Assigned Emotion']}")
                st.markdown("---")
    else:
        st.warning(f"No static file found at `{filename}`.")

elif page == "Visualization":
    st.title("Visualization (Demo Mode)")
    st.info("⚠ This is a static demo. Dataset will be loaded from predefined files.")
    dataset_choice = st.selectbox("Choose a dataset", ["reddit", "twitter"])
    filename = f"data/sentiment_{dataset_choice}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        st.session_state.topic_emotions_df = df
    else:
        st.warning(f"No static file found at `{filename}`.")

    if not st.session_state.topic_emotions_df.empty:
        st.subheader("Visualization of Topics and Emotions")
        df = st.session_state.topic_emotions_df
        emotion_color_map = {
            "Good (Positive)": "#2ECC71",
            "Fear (Negative)": "#E67E22",
            "Happy (Positive)": "#58D68D",
            "Disgust (Negative)": "#C0392B",
            "Sadness (Negative)": "#5DADE2",
            "Surprise (Neutral)": "#F4D03F"
        }

        topic_counts = df["Assigned Topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        num_topics = len(topic_counts)
        color_palette = sns.color_palette("husl", num_topics)
        topic_counts["Color"] = [to_hex(color_palette[i]) for i in range(num_topics)]

        bar_chart = alt.Chart(topic_counts).mark_bar().encode(
            x=alt.X("Topic:N", title="Topics"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Color:N", scale=alt.Scale(domain=topic_counts["Color"].tolist(), range=topic_counts["Color"].tolist()), legend=None),
            tooltip=["Topic", "Count"]
        ).properties(title="Distribution of Topics", width=350, height=300)

        overall_emotion_counts = df["Assigned Emotion"].value_counts().reset_index()
        overall_emotion_counts.columns = ["Emotion", "Count"]

        pie_chart = alt.Chart(overall_emotion_counts).mark_arc().encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color("Emotion:N", scale=alt.Scale(domain=list(emotion_color_map.keys()), range=list(emotion_color_map.values()))),
            tooltip=["Emotion", "Count"]
        ).properties(title="Overall Emotion Distribution", width=350, height=300)

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(bar_chart, use_container_width=True)
        with col2:
            st.altair_chart(pie_chart, use_container_width=True)

        st.subheader("Radar Chart & Word Cloud")
        radar_data = df.groupby(["Assigned Topic", "Assigned Emotion"]).size().unstack(fill_value=0)
        categories = list(radar_data.columns)
        topics = list(radar_data.index)
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        topic_color_map = dict(zip(topic_counts["Topic"], topic_counts["Color"]))

        for topic in topics:
            values = radar_data.loc[topic].values.tolist()
            values += values[:1]
            color = topic_color_map.get(topic, "#333333")
            ax_radar.plot(angles, values, color=color, linewidth=1.5)
            ax_radar.fill(angles, values, color=color, alpha=0.2)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=7, fontweight='bold', color="black", ha='center')
        ax_radar.set_yticklabels([])
        ax_radar.set_title("Emotion by Topic", fontsize=10, fontweight='bold', pad=10)
        ax_radar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7, labelspacing=0.3, ncol=2, frameon=False)
        plt.tight_layout()

        topic_text = " ".join(df["Assigned Topic"])
        fig_wc, ax_wc = plt.subplots(figsize=(4, 4), dpi=150)
        wordcloud_topic = WordCloud(background_color="white", colormap="plasma", width=600, height=300, max_words=100, prefer_horizontal=1.0).generate(topic_text)
        ax_wc.imshow(wordcloud_topic, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Most Frequent Words in Topics", fontsize=10)

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(fig_radar)
        with col4:
            st.pyplot(fig_wc)

        st.subheader("Heatmap of Emotion-Topic Correlation")
        fig, ax = plt.subplots(figsize=(5, 3))
        emotion_counts = df.groupby("Assigned Topic")["Assigned Emotion"].value_counts().reset_index(name="Count")
        sns.heatmap(
            emotion_counts.pivot(index="Assigned Topic", columns="Assigned Emotion", values="Count").fillna(0),
            cmap="coolwarm",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Emotion-Topic Heatmap")
        st.pyplot(fig)
