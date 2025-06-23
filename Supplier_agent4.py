# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:40:26 2025

@author: Rahul.Bhattacharyya
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:20:41 2025
@author: Rahul.Bhattacharyya
"""

import os
import re
import requests
import subprocess
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import truststore
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

# Required for SSL verification
truststore.inject_into_ssl()

class SupplierAgentPlugin:
    FILTER_BY_PARTS = False
    PART_KEYWORDS = [
        "fan blades", "acoustic panels", "composites", "carbon fiber",
        "nacelle", "engine casing", "thermoset"
    ]

    def __init__(self):
        self.impact_db = {}

    def _filter_news(self, texts):
        pattern = re.compile(r"(launch|shutdown|acquisition|merger|expanding|innovation|R&D|plant|investment)", re.IGNORECASE)
        base_filtered = [t for t in texts if pattern.search(t)]
        if self.FILTER_BY_PARTS:
            part_pattern = re.compile(r"|".join(re.escape(p) for p in self.PART_KEYWORDS), re.IGNORECASE)
            return [t for t in base_filtered if part_pattern.search(t)]
        return base_filtered

 

    @kernel_function(name="scrape_supplier_news", description="Safely scrapes news from supplier website with depth limit.")
    async def scrape_supplier_news(self, kernel, arguments: KernelArguments) -> list:
        start_url = arguments["url"]
        visited = set()
        news_items = []
    
        def is_valid(href):
            if not href:
                return False
            href = href.lower()
            return not (href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"))
    
        def should_visit(link):
            return link not in visited and start_url in link
    
        def crawl(url, depth=0, max_depth=2):
            if depth > max_depth or url in visited:
                return
            visited.add(url)
    
            try:
                response = requests.get(url, verify=False)
                if "text/html" not in response.headers.get("Content-Type", ""):
                    return
            except Exception as e:
                print(f"Failed to access {url}: {e}")
                return
    
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Extract news paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30)
            if text:
                news_items.append(text)
    
            # Follow links recursively
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if is_valid(href):
                    full_link = urljoin(url, href)
                    if should_visit(full_link):
                        crawl(full_link, depth + 1)
    
        crawl(start_url)
        return news_items


    @kernel_function(name="scrape_twitter_news", description="Scrape and filter supplier news from Twitter.")
    async def scrape_twitter_news(self, kernel: Kernel, arguments: KernelArguments) -> list:
        handle = arguments["handle"]
        max_results = arguments.get("max_results", 10)
        query = f"snscrape --max-results {max_results} twitter-search '{handle} since:2024-01-01'"
        try:
            result = subprocess.run(query, shell=True, capture_output=True, text=True)
            tweets = result.stdout.splitlines()
            return self._filter_news(tweets)
        except Exception as e:
            return [f"Error scraping Twitter: {e}"]

    @kernel_function(name="classify_news_intent", description="Classify supplier news intent.")
    async def classify_news_intent(self, kernel: Kernel, arguments: KernelArguments) -> str:
        service = kernel.get_service("gpt-4")
        news = arguments["news"]
        prompt = f"""
        Classify the intent of the following supplier news into one of these categories:
        - M&A
        - Innovation
        - Shutdown
        - Expansion
        - Other

        News: {news}
        """
        history = ChatHistory()
        history.add_user_message(prompt)
        
        # CORRECTED: Use proper method for chat completion
        response = await service.get_chat_message_contents(
            chat_history=history,
            settings=OpenAIPromptExecutionSettings()
        )
        return response[0].content.strip()


    '''
    @kernel_function(name="assess_impact", description="Assess impact of supplier news on Rolls Royce.")
    async def assess_impact(self, kernel: Kernel, arguments: KernelArguments) -> dict:
        service = kernel.get_service("gpt-4")
        label = arguments["intent"]
        supplier = arguments["supplier"]
        news = arguments["news"]
    
        prompt = f"""
        Given the following supplier news and intent label, explain how it could affect Rolls Royce’s supply chain.
        Then assign an impact level (Low, Medium, High).
        Try to estimate the delay (in weeks or months) compared to historical delivery timelines from similar events in aerospace if possible.
    
        Supplier: {supplier}
        Intent: {label}
        News: {news}
    
        Respond with:
        - Impact Level: <Low/Medium/High>
        - Reason: <Short explanation>
        - How it may affect Rolls Royce: <Effect and estimated delay>
        - Classify the sentiment of the news for Rolls Royce as one of: Positive, Negative, Neutral.

        """
    
        history = ChatHistory()
        history.add_user_message(prompt)
    
        response = await service.get_chat_message_contents(
            chat_history=history,
            settings=OpenAIPromptExecutionSettings()
        )
    
        explanation = response[0].content.strip()
    
        # Fallbacks and extraction using regex
        impact_data = {"impact": "Unknown", "reason": "", "rr_effect": "","sentiment":""}
        try:
            impact_match = re.search(r"(?i)impact level\s*[:\-]?\s*(\w+)", explanation)
            reason_match = re.search(r"(?i)reason\s*[:\-]?\s*(.+?)(?=\n|$)", explanation)
            effect_match = re.search(r"(?i)how.*?affect.*?[:\-]?\s*(.+?)(?=\n|$)", explanation)
    
            if impact_match:
                impact_data["impact"] = impact_match.group(1).strip()
            if reason_match:
                impact_data["reason"] = reason_match.group(1).strip()
            if effect_match:
                impact_data["rr_effect"] = effect_match.group(1).strip()
        except Exception as e:
            impact_data["reason"] = f"Parsing failed: {e}"
    
        return impact_data
    '''
    @kernel_function(name="assess_impact", description="Assess impact of supplier news on Rolls Royce.")
    async def assess_impact(self, kernel: Kernel, arguments: KernelArguments) -> dict:
        service = kernel.get_service("gpt-4")
        label = arguments["intent"]
        supplier = arguments["supplier"]
        news = arguments["news"]
    
        prompt = f"""
        Given the following supplier news and intent label, explain how it could affect Rolls Royce’s supply chain.
        Then assign an impact level (Low, Medium, High).
        Try to estimate the delay (in weeks or months) compared to historical delivery timelines from similar events in aerospace if possible.
        Classify the sentiment of the news for Rolls Royce as one of: Positive, Negative, Neutral.
        
    
        Supplier: {supplier}
        Intent: {label}
        News: {news}
    
        Respond with:
        - Impact Level: <Low/Medium/High>
        - Reason: <Short explanation>
        - Sentiment: <Positive/Negative/Neutral>
        """
    
        history = ChatHistory()
        history.add_user_message(prompt)
    
        response = await service.get_chat_message_contents(
            chat_history=history,
            settings=OpenAIPromptExecutionSettings()
        )
    
        explanation = response[0].content.strip()
    
        impact_data = {
            "impact": "Unknown",
            "reason": "",
            "rr_effect": "",
            "sentiment": ""
        }
    
        try:
            impact_match = re.search(r"(?i)impact level\s*[:\-]?\s*(\w+)", explanation)
            reason_match = re.search(r"(?i)reason\s*[:\-]?\s*(.+?)(?=\n|$)", explanation)
            effect_match = re.search(r"(?i)how.*?affect.*?[:\-]?\s*(.+?)(?=\n|$)", explanation)
            sentiment_match = re.search(r"(?i)^-?\s*sentiment\s*[:\-]?\s*(\w+)", explanation, re.MULTILINE)
    
            if impact_match:
                impact_data["impact"] = impact_match.group(1).strip()
            if reason_match:
                impact_data["reason"] = reason_match.group(1).strip()
            if effect_match:
                impact_data["rr_effect"] = effect_match.group(1).strip()
            if sentiment_match:
                impact_data["sentiment"] = sentiment_match.group(1).strip().capitalize()
        except Exception as e:
            impact_data["reason"] = f"Parsing failed: {e}"
    
        return impact_data



    @kernel_function(name="map_supply_chain_impact", description="Identify affected aerospace parts.")
    async def map_supply_chain_impact(self, kernel: Kernel, arguments: KernelArguments) -> str:
        service = kernel.get_service("gpt-4")
        news = arguments["news"]
        supplier = arguments["supplier"]
        prompt = f"""
        Based on the supplier news below, identify specific aerospace parts or subsystems that might be impacted.
    
        If the parts or subsystems are not directly mentioned, please infer the most likely impacted components based on
        common aerospace applications for the described material or technology. This may include examples such as:
        - airframe panels
        - wing skins
        - nacelles
        - engine casings
        - fairings
        - satellite structures
        - components in advanced air mobility (AAM) platforms
    
        Supplier: {supplier}
        News: {news}
    
        Respond with a concise bullet list of affected aerospace parts or subsystems if any:
        """
        history = ChatHistory()
        history.add_user_message(prompt)
        
        # CORRECTED: Use proper method for chat completion
        response = await service.get_chat_message_contents(
            chat_history=history,
            settings=OpenAIPromptExecutionSettings()
        )
        return response[0].content.strip()
    @kernel_function(name="scrape_macro_risks", description="Fetches global events that might affect supply chain delivery timelines.")
    async def scrape_macro_risks(self, kernel: Kernel, arguments: KernelArguments) -> str:
        supplier = arguments["supplier"]
        prompt = f"""
        Search for recent geopolitical or macroeconomic events from the past 2 weeks that could affect the delivery timeline or raw material supply for the aerospace supplier '{supplier}'.
        
        Summarize top 3-5 events and how they may impact supply chain timelines or costs.
        Focus on events like wars, trade sanctions, shipping delays, energy issues, etc.
        """
        
        chat = kernel.get_service("gpt-4")
        history = ChatHistory()
        history.add_user_message(prompt)
    
        response = await chat.get_chat_message_contents(
            chat_history=history,
            settings=OpenAIPromptExecutionSettings()
        )
    
        return response[0].content.strip()


# Set API key

'''
# CLI Runner
import pandas as pd

async def run_supplier_pipeline(url, handle, supplier):
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        service_id="gpt-4",
        ai_model_id="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    ))

    plugin = SupplierAgentPlugin()
    kernel.add_plugin(plugin, "supplier")

    context = KernelArguments()
    context["url"] = url
    context["handle"] = handle
    context["supplier"] = supplier

    scrape = kernel.get_function("supplier", "scrape_supplier_news")
    news_items_result = await scrape.invoke(kernel, context)
    news_items = news_items_result.value

    results = []

    for news in news_items:
        context["news"] = news

        classify = kernel.get_function("supplier", "classify_news_intent")
        intent_result = await classify.invoke(kernel, context)
        context["intent"] = intent_result.value

        assess = kernel.get_function("supplier", "assess_impact")
        impact_result = await assess.invoke(kernel, context)

        parts = kernel.get_function("supplier", "map_supply_chain_impact")
        partlist_result = await parts.invoke(kernel, context)

        impact = impact_result.value
        parts_affected = partlist_result.value

        print("\n📰 News:", news)
        print("📌 Intent:", context["intent"])
        print("⚠️ Impact:", impact)
        print("🔧 Affected Parts:", parts_affected)

        results.append({
            "supplier": supplier,
            "news": news,
            "intent": context["intent"],
            "impact_level": impact.get("impact", ""),
            "reason": impact.get("reason", ""),
            "rr_effect": impact.get("rr_effect", ""),
            "sentiment": impact.get("sentiment", ""),
            "affected_parts": parts_affected
        })

    # Save all results to a CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{supplier.lower()}_news_analysis.csv", index=False)
    print(f"\n✅ Results saved to {supplier.lower()}_news_analysis.csv")


# Run the pipeline
'''
import pandas as pd

async def run_supplier_pipeline(url, handle, supplier,streamlit_hook=True):
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        service_id="gpt-4",
        ai_model_id="gpt-4",
        api_key=st.secrets["OPENAI_API_KEY"]
    ))

    plugin = SupplierAgentPlugin()
    kernel.add_plugin(plugin, "supplier")

    context = KernelArguments()
    context["url"] = url
    context["handle"] = handle
    context["supplier"] = supplier

    # --- Scrape Website News ---
    if streamlit_hook:
        streamlit_hook.markdown("🔍 Scraping supplier website news...")
    scrape_site = kernel.get_function("supplier", "scrape_supplier_news")
    site_news_result = await scrape_site.invoke(kernel, context)
    site_news = site_news_result.value if site_news_result else []

    # --- Scrape Twitter News ---
    if streamlit_hook:
        streamlit_hook.markdown("🐦 Scraping supplier Twitter news...")
    scrape_twitter = kernel.get_function("supplier", "scrape_twitter_news")
    twitter_news_result = await scrape_twitter.invoke(kernel, context)
    twitter_news = twitter_news_result.value if twitter_news_result else []

    if streamlit_hook:
        streamlit_hook.markdown("🔍 scraping macroeconmic risks ...")
    macro_fn = kernel.get_function("supplier", "scrape_macro_risks")
    macro_summary = await macro_fn.invoke(kernel, context)

    # --- Combine both news sources ---
    all_news = site_news + twitter_news

    seen = set()
    unique_news = []
    for news in all_news:
        cleaned = news.strip().lower()
        if cleaned not in seen:
            seen.add(cleaned)
            unique_news.append(news)

    results = []

    for idx, news in enumerate(unique_news):
        context["news"] = news

        if streamlit_hook:
            streamlit_hook.markdown(f"🧠 Analyzing news item {idx + 1}/{len(unique_news)}")

        classify = kernel.get_function("supplier", "classify_news_intent")
        intent_result = await classify.invoke(kernel, context)
        context["intent"] = intent_result.value

        assess = kernel.get_function("supplier", "assess_impact")
        impact_result = await assess.invoke(kernel, context)

        parts = kernel.get_function("supplier", "map_supply_chain_impact")
        partlist_result = await parts.invoke(kernel, context)

        impact = impact_result.value
        parts_affected = partlist_result.value

        results.append({
            "supplier": supplier,
            "news": news,
            "intent": context["intent"],
            "impact_level": impact.get("impact", ""),
            "reason": impact.get("reason", ""),
            "rr_effect": impact.get("rr_effect", ""),
            "sentiment": impact.get("sentiment", ""),
            "affected_parts": parts_affected
        })

    df = pd.DataFrame(results)
    return df,macro_summary     

    #df.to_csv(f"{supplier.lower()}_news_analysis.csv", index=False)
    #print(f"\n✅ Results saved to {supplier.lower()}_news_analysis.csv")

'''
await run_supplier_pipeline(
     url="https://www.hexcel.com/News",
     handle="from:HexcelCorp",
     supplier="Hexcel"
 )

if __name__ == "__main__":
    asyncio.run(run_supplier_pipeline(
        url="https://www.hexcel.com/News",
        handle="from:HexcelCorp",
        supplier="Hexcel"
    ))
    
   
from semantic_kernel.connectors.ai.open_ai import OpenAIPromptExecutionSettings
import re

@kernel_function(name="assess_impact", description="Assess impact of supplier news on Rolls Royce.")
async def assess_impact(self, kernel: Kernel, arguments: KernelArguments) -> dict:
    service = kernel.get_service("gpt-4")
    label = arguments["intent"]
    supplier = arguments["supplier"]
    news = arguments["news"]

    prompt = f"""
    Given the following supplier news and intent label, explain how it could affect Rolls Royce’s supply chain.
    Then assign an impact level (Low, Medium, High).
    Try to estimate the delay (in weeks or months) compared to historical delivery timelines from similar events in aerospace if possible.

    Supplier: {supplier}
    Intent: {label}
    News: {news}

    Respond with:
    - Impact Level: <Low/Medium/High>
    - Reason: <Short explanation>
    - How it may affect Rolls Royce: <Effect and estimated delay>
    """

    history = ChatHistory()
    history.add_user_message(prompt)

    response = await service.get_chat_message_contents(
        chat_history=history,
        settings=OpenAIPromptExecutionSettings()
    )

    explanation = response[0].content.strip()

    # Fallbacks and extraction using regex
    impact_data = {"impact": "Unknown", "reason": "", "rr_effect": ""}
    try:
        impact_match = re.search(r"(?i)impact level\s*[:\-]?\s*(\w+)", explanation)
        reason_match = re.search(r"(?i)reason\s*[:\-]?\s*(.+?)(?=\n|$)", explanation)
        effect_match = re.search(r"(?i)how.*?affect.*?[:\-]?\s*(.+?)(?=\n|$)", explanation)

        if impact_match:
            impact_data["impact"] = impact_match.group(1).strip()
        if reason_match:
            impact_data["reason"] = reason_match.group(1).strip()
        if effect_match:
            impact_data["rr_effect"] = effect_match.group(1).strip()
    except Exception as e:
        impact_data["reason"] = f"Parsing failed: {e}"

    return impact_data

#import truststore
#truststore.inject_into_ssl()

import requests

url = "https://www.toraytac.com/news"
requests.get(url, verify=False)

print("Status:", response.status_code)
print("Headers:", response.headers)
'''
