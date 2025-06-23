# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 00:06:33 2025

@author: Rahul.Bhattacharyya
"""

# streamlit_supplier_news.py — Streamlit frontend for SupplierAgentPlugin

import streamlit as st
import asyncio
import pandas as pd
from Supplier_agent4 import run_supplier_pipeline

# Title and Description
st.set_page_config(page_title="Supplier News Analyzer", layout="wide")
st.title("📦 Supplier News Impact Analyzer")
st.markdown("Analyze supplier news impact on Rolls Royce with agentic AI.")

# Sidebar Inputs
st.sidebar.header("⚙️ Configuration")
supplier = st.sidebar.selectbox("Select Supplier", ["Hexcel"])
filter_by_parts = st.sidebar.checkbox("Filter by Aerospace Part Keywords", value=True)
keywords_input = st.sidebar.text_input("Additional Keywords (comma-separated)", "")

url = "https://www.hexcel.com/News"
handle = "from:HexcelCorp"

# Button to run the pipeline
if st.sidebar.button("🚀 Run Analysis"):
    st.info("⏳ Running agent pipeline...")

    # Chain of Thought Progress
    with st.expander("🔄 Chain of Thought Progress", expanded=True):
        chain_status = st.empty()
        step_progress = st.progress(0, text="Initializing...")

    # Save results container
    results_placeholder = st.empty()

    # Run async pipeline
    async def run():
        from Supplier_agent4 import SupplierAgentPlugin
        SupplierAgentPlugin.FILTER_BY_PARTS = filter_by_parts
        if keywords_input:
            SupplierAgentPlugin.PART_KEYWORDS = [k.strip() for k in keywords_input.split(",") if k.strip()]

        try:
            step_progress.progress(10, text="🔍 Scraping supplier website...")
            df,macro_summary = await run_supplier_pipeline(url=url, handle=handle, supplier=supplier,streamlit_hook=chain_status)
            step_progress.progress(90, text="📊 Preparing results...")

            if not df.empty:
                st.success("✅ Analysis completed!")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, file_name=f"{supplier.lower()}_news_analysis.csv", mime="text/csv")
            else:
                st.warning("No relevant news articles found.")

            step_progress.progress(100, text="✅ Done")
            
            # Show macroeconomic impact summary
            with st.expander("🌍 External Macroeconomic Risks", expanded=False):
                st.markdown(macro_summary)  
            
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")

    asyncio.run(run())
