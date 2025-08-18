import asyncio
from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv
import requests
import logging
import aiohttp
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the State
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    """
    ticker: str
    company_info: Optional[dict]
    historical_prices: Optional[pd.DataFrame]
    historical_analysis: Optional[str]
    news_summaries: Optional[List[str]]
    recommendation: Optional[dict]
    final_report: Optional[dict]
    mcp_session: ClientSession

# Helper function for OpenAI API calls with exponential backoff
async def call_openai_api_with_exponential_backoff(session, payload, max_retries=3):
    """Make OpenAI API call with exponential backoff."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_endpoint = os.getenv("OPENAI_ENDPOINT")
    # azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    # Determine if using Azure OpenAI or regular OpenAI
    # if openai_endpoint and "openai.azure.com" in openai_endpoint:
        # Azure OpenAI format
        # if not azure_deployment:
        #     logger.error("AZURE_DEPLOYMENT_NAME is required for Azure OpenAI")
        #     return None
        
        # Azure OpenAI URL format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2025-01-01-preview
        # url = f"{openai_endpoint}/openai/deployments/{azure_deployment}/chat/completions?api-version=2025-01-01-preview"
    url = openai_endpoint
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key  # Azure uses 'api-key' header
    }
    # else:
    #     # Regular OpenAI format
    #     url = "https://api.openai.com/v1/chat/completions"
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {openai_api_key}"
    #     }
    
    logger.info(f"Making API call to: {url}")
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"API Response Status: {response.status}")
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"OpenAI API error: {response.status} - {response_text}")
                        if attempt == max_retries - 1:
                            return None
                        await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"OpenAI API call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(2 ** attempt)
    
    return None

# 2. Define the Agent Nodes with corrected MCP Tool Calls

async def data_analysis_agent(state: AgentState) -> AgentState:
    """Data Analysis Agent: Fetches financial data and performs initial assessment."""
    logger.info(f"📊 Data Analysis Agent: Fetching and analyzing data for {state['ticker']}...")
    ticker = state['ticker']
    session = state['mcp_session']

    try:
        # Initialize state with empty values
        state['company_info'] = {}
        state['historical_prices'] = pd.DataFrame()
        state['historical_analysis'] = "No analysis available"

        # FIXED: Use call_tool instead of send_request
        result = await session.call_tool("get_company_data", {"ticker_symbol": ticker})
        
        if hasattr(result, 'content') and result.content:
            # Parse the result content
            if isinstance(result.content[0], dict):
                data = result.content[0]
            else:
                data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
            
            if not data.get('error'):
                state['company_info'] = data.get('info', {})
                
                historical_data_dict = data.get('historical_prices', {})
                historical_records = historical_data_dict.get('data', [])
                df = pd.DataFrame(historical_records) if historical_records else pd.DataFrame()
                state['historical_prices'] = df
                
                if not df.empty and 'Close' in df.columns:
                    df['5d_MA'] = df['Close'].rolling(window=5).mean()
                    latest_ma = df['5d_MA'].iloc[-1]
                    latest_close = df['Close'].iloc[-1]
                    analysis_text = f"The 5-day moving average is ${latest_ma:.2f}, while the latest closing price is ${latest_close:.2f}."
                    if latest_close > latest_ma:
                        analysis_text += " The recent price trend is positive."
                    else:
                        analysis_text += " The recent price trend is negative."
                    state['historical_analysis'] = analysis_text
            else:
                logger.error(f"Error in API response: {data.get('error')}")
        else:
            logger.error("No content received from MCP tool")
    
    except Exception as e:
        logger.error(f"❌ MCP tool call 'get_company_data' failed: {e}")
    
    return state

async def news_summarization_agent(state: AgentState) -> AgentState:
    """News Summarization Agent: Searches for news and generates summaries."""
    logger.info(f"📰 News Summarization Agent: Searching for news about {state['ticker']}")
    ticker = state['ticker']
    company_name = state.get('company_info', {}).get('shortName', ticker)
    session = state['mcp_session']

    try:
        state['news_summaries'] = []  # Initialize with empty list
        
        # FIXED: Use call_tool instead of send_request
        result = await session.call_tool("search_for_news", {"query": company_name})
        
        news_articles = []
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content[0], dict):
                news_articles = result.content[0]
            else:
                news_articles = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else []
        
        if not news_articles:
            logger.info("No news articles found.")
            state['news_summaries'] = ["No recent news found."]
            return state

        summaries = []
        logger.info(f"Found {len(news_articles)} articles. Processing content...")

        for article in news_articles[:3]:
            url = article.get('url', 'N/A')
            try:
                # FIXED: Use call_tool for web scraping
                soup_result = await session.call_tool("get_soup_from_url", {"url": url})
                
                page_content = {}
                if hasattr(soup_result, 'content') and soup_result.content:
                    if isinstance(soup_result.content[0], dict):
                        page_content = soup_result.content[0]
                    else:
                        page_content = json.loads(soup_result.content[0].text) if hasattr(soup_result.content[0], 'text') else {}
                
                article_text = page_content.get("text", "")

                if article_text:
                    prompt = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that summarizes financial news."},
                            {"role": "user", "content": f"Summarize the following news article text concisely for a financial report. Focus on key financial details, company performance, and market sentiment: {article_text[:4000]}"}
                        ],
                        "max_tokens": 150,
                        "temperature": 0.3
                    }
                    
                    try:
                        response = await call_openai_api_with_exponential_backoff(session, prompt)
                        if response and response.get('choices'):
                            summary_text = response['choices'][0]['message']['content']
                            summaries.append(f"Summary from {url}: {summary_text}")
                        else:
                            summaries.append(f"Could not generate summary for article at {url}.")
                    except Exception as e:
                        logger.error(f"LLM summarization failed for {url}: {e}")
                        summaries.append(f"Error summarizing article at {url}.")
                else:
                    summaries.append(f"No content found for article at {url}.")
            except Exception as e:
                logger.error(f"Error processing article at {url}: {e}")
                summaries.append(f"Error processing article at {url}.")
        
        state['news_summaries'] = summaries
        logger.info("News summarization complete.")
    
    except Exception as e:
        logger.error(f"Error during news summarization: {e}")
        state['news_summaries'] = ["Error during news retrieval and summarization."]
    
    return state

async def recommendation_agent(state: AgentState) -> AgentState:
    """
    Recommendation Agent: Synthesizes information to provide an investment recommendation.
    """
    logger.info("💡 Recommendation Agent: Synthesizing data for an investment recommendation...")
    company_info = state.get('company_info', {})
    historical_analysis = state.get('historical_analysis', "No historical analysis available.")
    news_summaries = state.get('news_summaries', ["No news available."])
    session = state['mcp_session']

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        state['recommendation'] = {"recommendation": "Hold", "rationale": "API key not found."}
        return state

    # Create a structured prompt for the LLM
    prompt_text = f"""
    Based on the following data, act as a senior portfolio manager and provide a clear, actionable investment recommendation (e.g., 'Buy', 'Hold', 'Sell'). Justify your recommendation with a brief rationale, citing the provided financial analysis and news summaries.
    
    Financial Analysis: {historical_analysis}
    Company Info: {json.dumps(company_info)}
    News Summaries: {json.dumps(news_summaries)}
    
    Format your response as a JSON object with 'recommendation' and 'rationale' keys.
    """
    
    try:
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates investment recommendations in JSON format."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
        
        response = await call_openai_api_with_exponential_backoff(session, payload)

        recommendation_data = {"recommendation": "Hold", "rationale": "Could not generate a specific recommendation."}
        
        if response and response.get('choices'):
            json_text = response['choices'][0]['message']['content']
            logger.info(f"LLM Response: {json_text}")
            try:
                recommendation_data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {json_text}")
                # Try to extract recommendation manually if JSON parsing fails
                if "buy" in json_text.lower():
                    recommendation_data = {"recommendation": "Buy", "rationale": "Based on positive analysis"}
                elif "sell" in json_text.lower():
                    recommendation_data = {"recommendation": "Sell", "rationale": "Based on negative analysis"}
                else:
                    recommendation_data = {"recommendation": "Hold", "rationale": "Neutral analysis"}
        
        state['recommendation'] = recommendation_data
        logger.info("✅ Recommendation generation complete.")

    except Exception as e:
        logger.error(f"Error during recommendation generation: {e}")
        state['recommendation'] = {"recommendation": "Hold", "rationale": "Error during generation."}

    return state

async def report_generation_agent(state: AgentState) -> AgentState:
    """
    Report Generation Agent: Assembles the final, comprehensive report.
    """
    logger.info("✍️ Report Generation Agent: Assembling the final report...")
    company_info = state.get('company_info', {})
    historical_prices = state.get('historical_prices', pd.DataFrame())
    historical_analysis = state.get('historical_analysis', "No analysis available.")
    news_summaries = state.get('news_summaries', ["No news available."])
    recommendation = state.get('recommendation', {})
    ticker = state.get('ticker')
    session = state['mcp_session']

    title = f"Financial Health and News Report for {company_info.get('shortName', ticker)}"
    current_price = company_info.get('currentPrice', 'N/A')
    market_cap = company_info.get('marketCap', 'N/A')
    pe_ratio = company_info.get('trailingPE', 'N/A')
    sector = company_info.get('sector', 'N/A')
    
    pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else str(pe_ratio)
    financial_summary = (
        f"**Current Stock Price:** ${current_price}\n"
        f"**Market Capitalization:** ${market_cap:,}\n"
        f"**P/E Ratio:** {pe_ratio_str}\n"
        f"**Sector:** {sector}"
    )

    historical_table_data = {'columns': [], 'data': []}
    if not historical_prices.empty:
        # Check what columns are actually available
        available_columns = historical_prices.columns.tolist()
        logger.info(f"Available columns in historical_prices: {available_columns}")
        
        # Try to find date and close columns with different possible names
        date_col = None
        close_col = None
        
        for col in available_columns:
            if col.lower() in ['date', 'datetime', 'timestamp']:
                date_col = col
            elif col.lower() in ['close', 'closing', 'closing_price']:
                close_col = col
        
        # If we have the required columns, create the table data
        if close_col:
            if date_col:
                columns = [date_col, close_col]
                data_records = historical_prices[[date_col, close_col]].tail(5).to_dict('records')
            else:
                # If no date column, just use Close prices with index as date
                columns = ['Date', close_col]
                df_with_date = historical_prices[[close_col]].tail(5).copy()
                df_with_date['Date'] = df_with_date.index.strftime('%Y-%m-%d') if hasattr(df_with_date.index, 'strftime') else df_with_date.index.astype(str)
                data_records = df_with_date[['Date', close_col]].to_dict('records')
            
            historical_table_data = {'columns': columns, 'data': data_records}
        else:
            logger.warning("No 'Close' column found in historical data")
        
    report_data = {
        "title": title,
        "sections": [
            {
                "heading": "Executive Summary",
                "body_text": (
                    f"**Recommendation:** {recommendation.get('recommendation', 'N/A')}\n\n"
                    f"**Rationale:** {recommendation.get('rationale', 'N/A')}"
                )
            },
            {
                "heading": "Company Overview",
                "body_text": f"This report provides a summary of the financial health and recent news for {company_info.get('longName', ticker)}. The company operates in the {sector} sector and has a market capitalization of ${market_cap:,}."
            },
            {
                "heading": "Key Financial Metrics & Analysis",
                "body_text": f"{financial_summary}\n\n**Historical Analysis:** {historical_analysis}"
            },
            {
                "heading": "Recent News Summary",
                "body_text": "\n\n".join([f"- {s}" for s in news_summaries])
            }
        ],
        "footer": "Report generated by the Agentic AI Financial Analyst"
    }

    try:
        filename = f"{ticker}_financial_report.docx"
        
        # FIXED: Use call_tool for report generation
        result = await session.call_tool("create_professional_report", {
            "filename": filename,
            "report_data": report_data,
            "table_data": historical_table_data
        })
        
        if hasattr(result, 'content') and result.content:
            response_data = {}
            if isinstance(result.content[0], dict):
                response_data = result.content[0]
            else:
                response_data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
            
            if response_data.get('error'):
                raise Exception(f"MCP error: {response_data['error']}")
        
    except Exception as e:
        logger.error(f"❌ MCP tool call 'create_professional_report' failed: {e}")
        state['final_report'] = {"error": "Report generation failed."}
        return state
    
    state['final_report'] = report_data
    logger.info("✅ Report generation complete.")
    return state


# 3. Build the LangGraph Workflow
def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("data_analysis", data_analysis_agent)
    workflow.add_node("news_summarization", news_summarization_agent)
    workflow.add_node("recommendation", recommendation_agent)
    workflow.add_node("report_generation", report_generation_agent)

    workflow.set_entry_point("data_analysis")
    workflow.add_edge("data_analysis", "news_summarization")
    workflow.add_edge("news_summarization", "recommendation")
    workflow.add_edge("recommendation", "report_generation")
    workflow.set_finish_point("report_generation")
    
    app = workflow.compile()
    return app

# 4. Main Execution Loop
async def main():
    # Define server parameters - FIXED: Use correct filename
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]  # Changed from mcp_server.py to match your actual filename
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                graph_app = build_graph()
                
                initial_state: AgentState = {
                    "ticker": "GOOG",
                    "company_info": None,
                    "historical_prices": None,
                    "historical_analysis": None,
                    "news_summaries": None,
                    "recommendation": None,
                    "final_report": None,
                    "mcp_session": session
                }
                
                logger.info("🚀 Starting the multi-agent financial analysis pipeline...")
                final_state = await graph_app.ainvoke(initial_state)
                logger.info("✅ Pipeline finished. Report has been generated.")
                logger.debug("Final State: %s", final_state)
    except TimeoutError:
        logger.error("Timeout while starting MCP server")
        raise
    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())