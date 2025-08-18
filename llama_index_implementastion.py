import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv
import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow.context import Context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalysisState:
    """Shared state for the financial analysis pipeline."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.company_info: Optional[dict] = None
        self.historical_prices: Optional[pd.DataFrame] = None
        self.historical_analysis: Optional[str] = None
        self.news_summaries: Optional[List[str]] = None
        self.recommendation: Optional[dict] = None
        self.final_report: Optional[dict] = None
        self.mcp_session: Optional[ClientSession] = None

# Global state instance
analysis_state = None

async def call_openai_api_with_exponential_backoff(payload, max_retries=3):
    """Make OpenAI API call with exponential backoff."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_endpoint = os.getenv("OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    if openai_endpoint and "openai.azure.com" in openai_endpoint:
        url = f"{openai_endpoint}/openai/deployments/{azure_deployment}/chat/completions?api-version=2025-01-01-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": openai_api_key
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"OpenAI API error: {response.status}")
                        if attempt == max_retries - 1:
                            return None
                        await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"OpenAI API call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(2 ** attempt)
    
    return None

# Workflow Events
class DataAnalysisEvent(Event):
    ticker: str

class NewsAnalysisEvent(Event):
    company_name: str

class RecommendationEvent(Event):
    analysis_complete: bool

class ReportGenerationEvent(Event):
    recommendation_complete: bool

class FinancialAnalysisWorkflow(Workflow):
    """LlamaIndex Workflow for Financial Analysis"""

    def __init__(self, timeout: float = 300.0, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)

    @step
    async def data_analysis_step(
        self, ctx: Context, ev: StartEvent
    ) -> DataAnalysisEvent:
        """Step 1: Analyze financial data"""
        logger.info(f"📊 Data Analysis Step: Fetching data for {analysis_state.ticker}")
        
        try:
            # Initialize state with empty values
            analysis_state.company_info = {}
            analysis_state.historical_prices = pd.DataFrame()
            analysis_state.historical_analysis = "No analysis available"

            result = await analysis_state.mcp_session.call_tool("get_company_data", {"ticker_symbol": analysis_state.ticker})
            
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content[0], dict):
                    data = result.content[0]
                else:
                    data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
                
                if not data.get('error'):
                    analysis_state.company_info = data.get('info', {})
                    
                    historical_data_dict = data.get('historical_prices', {})
                    historical_records = historical_data_dict.get('data', [])
                    df = pd.DataFrame(historical_records) if historical_records else pd.DataFrame()
                    analysis_state.historical_prices = df
                    
                    if not df.empty and 'Close' in df.columns:
                        df['5d_MA'] = df['Close'].rolling(window=5).mean()
                        latest_ma = df['5d_MA'].iloc[-1]
                        latest_close = df['Close'].iloc[-1]
                        analysis_text = f"The 5-day moving average is ${latest_ma:.2f}, while the latest closing price is ${latest_close:.2f}."
                        if latest_close > latest_ma:
                            analysis_text += " The recent price trend is positive."
                        else:
                            analysis_text += " The recent price trend is negative."
                        analysis_state.historical_analysis = analysis_text
                    
                    logger.info("✅ Data analysis completed successfully")
                    return DataAnalysisEvent(ticker=analysis_state.ticker)
                else:
                    logger.error(f"Error in data fetch: {data.get('error')}")
            
            logger.warning("No valid data received")
            return DataAnalysisEvent(ticker=analysis_state.ticker)
            
        except Exception as e:
            logger.error(f"Error in data analysis step: {e}")
            return DataAnalysisEvent(ticker=analysis_state.ticker)

    @step
    async def news_analysis_step(
        self, ctx: Context, ev: DataAnalysisEvent
    ) -> NewsAnalysisEvent:
        """Step 2: Analyze financial news"""
        company_name = analysis_state.company_info.get('shortName', analysis_state.ticker)
        logger.info(f"📰 News Analysis Step: Searching news for {company_name}")
        
        try:
            analysis_state.news_summaries = []
            
            result = await analysis_state.mcp_session.call_tool("search_for_news", {"query": company_name})
            news_articles = []
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content[0], dict):
                    news_articles = result.content[0] if isinstance(result.content[0], list) else []
                else:
                    news_articles = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else []
            
            if not news_articles:
                analysis_state.news_summaries = ["No recent news found."]
                logger.info("No news articles found")
                return NewsAnalysisEvent(company_name=company_name)

            summaries = []
            for article in news_articles[:3]:
                url = article.get('url', 'N/A')
                try:
                    soup_result = await analysis_state.mcp_session.call_tool("get_soup_from_url", {"url": url})
                    
                    page_content = {}
                    if hasattr(soup_result, 'content') and soup_result.content:
                        if isinstance(soup_result.content[0], dict):
                            page_content = soup_result.content[0]
                        else:
                            page_content = json.loads(soup_result.content[0].text) if hasattr(soup_result.content[0], 'text') else {}
                    
                    article_text = page_content.get("text", "")

                    if article_text:
                        payload = {
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that summarizes financial news."},
                                {"role": "user", "content": f"Summarize the following news article text concisely for a financial report. Focus on key financial details, company performance, and market sentiment: {article_text[:4000]}"}
                            ],
                            "max_tokens": 150,
                            "temperature": 0.3
                        }
                        
                        response = await call_openai_api_with_exponential_backoff(payload)
                        if response and response.get('choices'):
                            summary_text = response['choices'][0]['message']['content']
                            summaries.append(f"Summary from {url}: {summary_text}")
                        else:
                            summaries.append(f"Could not generate summary for article at {url}.")
                    else:
                        summaries.append(f"No content found for article at {url}.")
                except Exception as e:
                    summaries.append(f"Error processing article at {url}: {str(e)}")
            
            analysis_state.news_summaries = summaries
            logger.info("✅ News analysis completed successfully")
            return NewsAnalysisEvent(company_name=company_name)
            
        except Exception as e:
            logger.error(f"Error in news analysis step: {e}")
            analysis_state.news_summaries = ["Error during news retrieval and summarization."]
            return NewsAnalysisEvent(company_name=company_name)

    @step
    async def recommendation_step(
        self, ctx: Context, ev: NewsAnalysisEvent
    ) -> RecommendationEvent:
        """Step 3: Generate investment recommendation"""
        logger.info("💡 Recommendation Step: Generating investment recommendation")
        
        try:
            company_info = analysis_state.company_info or {}
            historical_analysis = analysis_state.historical_analysis or "No historical analysis available."
            news_summaries = analysis_state.news_summaries or ["No news available."]

            prompt_text = f"""
            Based on the following data, act as a senior portfolio manager and provide a clear, actionable investment recommendation (e.g., 'Buy', 'Hold', 'Sell'). Justify your recommendation with a brief rationale, citing the provided financial analysis and news summaries.
            
            Financial Analysis: {historical_analysis}
            Company Info: {json.dumps(company_info)}
            News Summaries: {json.dumps(news_summaries)}
            
            Format your response as a JSON object with 'recommendation' and 'rationale' keys.
            """
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates investment recommendations in JSON format."},
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": 200,
                "response_format": {"type": "json_object"}
            }
            
            response = await call_openai_api_with_exponential_backoff(payload)

            recommendation_data = {"recommendation": "Hold", "rationale": "Could not generate a specific recommendation."}
            
            if response and response.get('choices'):
                json_text = response['choices'][0]['message']['content']
                try:
                    recommendation_data = json.loads(json_text)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {json_text}")
                    if "buy" in json_text.lower():
                        recommendation_data = {"recommendation": "Buy", "rationale": "Based on positive analysis"}
                    elif "sell" in json_text.lower():
                        recommendation_data = {"recommendation": "Sell", "rationale": "Based on negative analysis"}
            
            analysis_state.recommendation = recommendation_data
            logger.info("✅ Recommendation generation completed successfully")
            return RecommendationEvent(analysis_complete=True)
            
        except Exception as e:
            logger.error(f"Error in recommendation step: {e}")
            analysis_state.recommendation = {"recommendation": "Hold", "rationale": "Error during generation."}
            return RecommendationEvent(analysis_complete=True)

    @step
    async def report_generation_step(
        self, ctx: Context, ev: RecommendationEvent
    ) -> StopEvent:
        """Step 4: Generate final report"""
        logger.info("✍️ Report Generation Step: Creating final report")
        
        try:
            company_info = analysis_state.company_info or {}
            historical_prices = analysis_state.historical_prices or pd.DataFrame()
            historical_analysis = analysis_state.historical_analysis or "No analysis available."
            news_summaries = analysis_state.news_summaries or ["No news available."]
            recommendation = analysis_state.recommendation or {}
            ticker = analysis_state.ticker

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

            # Handle historical table data
            historical_table_data = {'columns': [], 'data': []}
            if not historical_prices.empty:
                available_columns = historical_prices.columns.tolist()
                close_col = None
                
                for col in available_columns:
                    if col.lower() in ['close', 'closing', 'closing_price']:
                        close_col = col
                        break
                
                if close_col:
                    columns = ['Date', close_col]
                    df_with_date = historical_prices[[close_col]].tail(5).copy()
                    df_with_date['Date'] = df_with_date.index.strftime('%Y-%m-%d') if hasattr(df_with_date.index, 'strftime') else df_with_date.index.astype(str)
                    data_records = df_with_date[['Date', close_col]].to_dict('records')
                    historical_table_data = {'columns': columns, 'data': data_records}
                
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
                "footer": "Report generated by LlamaIndex Agentic AI Financial Analyst"
            }

            filename = f"{ticker}_financial_report_llamaindex.docx"
            
            result = await analysis_state.mcp_session.call_tool("create_professional_report", {
                "filename": filename,
                "report_data": report_data,
                "table_data": historical_table_data
            })
            
            response_data = {}
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content[0], dict):
                    response_data = result.content[0]
                else:
                    response_data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
            
            if response_data.get('status') == 'success' or not response_data.get('error'):
                analysis_state.final_report = report_data
                logger.info("✅ Report generation completed successfully")
                return StopEvent(result=f"Report successfully generated: {filename}")
            else:
                logger.error(f"Report generation failed: {response_data.get('message', 'Unknown error')}")
                return StopEvent(result=f"Report generation failed: {response_data.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error in report generation step: {e}")
            return StopEvent(result=f"Error generating report: {str(e)}")

# Tool Functions for LlamaIndex Agents (Alternative approach using ReActAgent)
def get_company_data_tool(ticker_symbol: str) -> str:
    """Fetch company financial data and stock prices."""
    async def _fetch_data():
        try:
            result = await analysis_state.mcp_session.call_tool("get_company_data", {"ticker_symbol": ticker_symbol})
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content[0], dict):
                    data = result.content[0]
                else:
                    data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
                
                if not data.get('error'):
                    analysis_state.company_info = data.get('info', {})
                    
                    historical_data_dict = data.get('historical_prices', {})
                    historical_records = historical_data_dict.get('data', [])
                    df = pd.DataFrame(historical_records) if historical_records else pd.DataFrame()
                    analysis_state.historical_prices = df
                    
                    if not df.empty and 'Close' in df.columns:
                        df['5d_MA'] = df['Close'].rolling(window=5).mean()
                        latest_ma = df['5d_MA'].iloc[-1]
                        latest_close = df['Close'].iloc[-1]
                        analysis_text = f"The 5-day moving average is ${latest_ma:.2f}, while the latest closing price is ${latest_close:.2f}."
                        if latest_close > latest_ma:
                            analysis_text += " The recent price trend is positive."
                        else:
                            analysis_text += " The recent price trend is negative."
                        analysis_state.historical_analysis = analysis_text
                    
                    return f"Successfully fetched data for {ticker_symbol}. Company: {analysis_state.company_info.get('shortName', 'N/A')}. Current Price: ${analysis_state.company_info.get('currentPrice', 'N/A')}. Analysis: {analysis_state.historical_analysis}"
                else:
                    return f"Error fetching data: {data.get('error')}"
            return "No data received from MCP tool"
        except Exception as e:
            return f"Error fetching company data: {str(e)}"
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(_fetch_data())
    loop.close()
    return result

def search_and_summarize_news_tool(company_name: str) -> str:
    """Search and summarize financial news for a company."""
    async def _fetch_news():
        try:
            analysis_state.news_summaries = []
            
            result = await analysis_state.mcp_session.call_tool("search_for_news", {"query": company_name})
            news_articles = []
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content[0], dict):
                    news_articles = result.content[0] if isinstance(result.content[0], list) else []
                else:
                    news_articles = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else []
            
            if not news_articles:
                analysis_state.news_summaries = ["No recent news found."]
                return "No news articles found."

            summaries = []
            for article in news_articles[:3]:
                url = article.get('url', 'N/A')
                try:
                    soup_result = await analysis_state.mcp_session.call_tool("get_soup_from_url", {"url": url})
                    
                    page_content = {}
                    if hasattr(soup_result, 'content') and soup_result.content:
                        if isinstance(soup_result.content[0], dict):
                            page_content = soup_result.content[0]
                        else:
                            page_content = json.loads(soup_result.content[0].text) if hasattr(soup_result.content[0], 'text') else {}
                    
                    article_text = page_content.get("text", "")

                    if article_text:
                        payload = {
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that summarizes financial news."},
                                {"role": "user", "content": f"Summarize the following news article text concisely for a financial report. Focus on key financial details, company performance, and market sentiment: {article_text[:4000]}"}
                            ],
                            "max_tokens": 150,
                            "temperature": 0.3
                        }
                        
                        response = await call_openai_api_with_exponential_backoff(payload)
                        if response and response.get('choices'):
                            summary_text = response['choices'][0]['message']['content']
                            summaries.append(f"Summary from {url}: {summary_text}")
                        else:
                            summaries.append(f"Could not generate summary for article at {url}.")
                    else:
                        summaries.append(f"No content found for article at {url}.")
                except Exception as e:
                    summaries.append(f"Error processing article at {url}: {str(e)}")
            
            analysis_state.news_summaries = summaries
            return f"Found and summarized {len(summaries)} news articles. Key insights: {'; '.join(summaries[:2])}"
        except Exception as e:
            analysis_state.news_summaries = ["Error during news retrieval and summarization."]
            return f"Error searching news: {str(e)}"
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(_fetch_news())
    loop.close()
    return result

async def main_workflow_approach():
    """Main function using LlamaIndex Workflow approach."""
    global analysis_state
    
    # Initialize MCP connection
    server_params = StdioServerParameters(
        command="python",
        args=["asset-mgmt-server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Initialize shared state
                analysis_state = FinancialAnalysisState("GOOG")
                analysis_state.mcp_session = session
                
                # Configure LlamaIndex LLM
                if os.getenv("OPENAI_ENDPOINT") and "openai.azure.com" in os.getenv("OPENAI_ENDPOINT"):
                    llm = AzureOpenAI(
                        model="gpt-4o",
                        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                        api_key=os.getenv("OPENAI_API_KEY"),
                        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                        api_version="2025-01-01-preview",
                    )
                else:
                    llm = OpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                
                Settings.llm = llm
                
                # Create and run workflow
                workflow = FinancialAnalysisWorkflow(timeout=300.0, verbose=True)
                
                logger.info("🚀 Starting LlamaIndex Workflow multi-agent financial analysis pipeline...")
                
                result = await workflow.run()
                
                logger.info("✅ LlamaIndex Workflow pipeline finished.")
                logger.info(f"Final Result: {result}")

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}")
        raise

async def main_agent_approach():
    """Main function using LlamaIndex ReActAgent approach."""
    global analysis_state
    
    # Initialize MCP connection
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Initialize shared state
                analysis_state = FinancialAnalysisState("GOOG")
                analysis_state.mcp_session = session
                
                # Configure LlamaIndex LLM
                if os.getenv("OPENAI_ENDPOINT") and "openai.azure.com" in os.getenv("OPENAI_ENDPOINT"):
                    llm = AzureOpenAI(
                        model="gpt-4o",
                        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                        api_key=os.getenv("OPENAI_API_KEY"),
                        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                        api_version="2025-01-01-preview",
                    )
                else:
                    llm = OpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                
                Settings.llm = llm
                
                # Create tools
                company_data_tool = FunctionTool.from_defaults(fn=get_company_data_tool)
                news_tool = FunctionTool.from_defaults(fn=search_and_summarize_news_tool)
                
                tools = [company_data_tool, news_tool]
                
                # Create ReAct Agent
                agent = ReActAgent.from_tools(
                    tools=tools,
                    llm=llm,
                    verbose=True
                )
                
                logger.info("🚀 Starting LlamaIndex ReActAgent multi-agent financial analysis pipeline...")
                
                # Step 1: Data Analysis
                logger.info("Step 1: Fetching company data...")
                data_response = agent.chat(f"Get company data for {analysis_state.ticker} and analyze the financial metrics and stock performance")
                logger.info(f"Data Analysis Response: {data_response}")
                
                # Step 2: News Analysis
                logger.info("Step 2: Searching and analyzing news...")
                company_name = analysis_state.company_info.get('shortName', analysis_state.ticker) if analysis_state.company_info else analysis_state.ticker
                news_response = agent.chat(f"Search and summarize recent financial news for {company_name}")
                logger.info(f"News Analysis Response: {news_response}")
                
                # Step 3: Generate Recommendation
                logger.info("Step 3: Generating investment recommendation...")
                recommendation_prompt = f"""
                Based on the financial data and news analysis completed, provide an investment recommendation for {analysis_state.ticker}.
                
                Consider:
                - Financial metrics and stock performance trends
                - Recent news sentiment and market developments
                - Overall risk/reward profile
                
                Provide a clear recommendation (Buy, Hold, or Sell) with detailed rationale.
                """
                
                recommendation_response = agent.chat(recommendation_prompt)
                logger.info(f"Investment Recommendation: {recommendation_response}")
                
                # Step 4: Generate Report (manual since it requires MCP tool call)
                logger.info("Step 4: Generating final report...")
                
                # Extract recommendation from agent response
                rec_text = str(recommendation_response)
                if "buy" in rec_text.lower() and "hold" not in rec_text.lower():
                    recommendation = "Buy"
                elif "sell" in rec_text.lower():
                    recommendation = "Sell"
                else:
                    recommendation = "Hold"
                
                analysis_state.recommendation = {
                    "recommendation": recommendation,
                    "rationale": rec_text
                }
                
                # Generate final report
                async def _generate_final_report():
                    try:
                        company_info = analysis_state.company_info or {}
                        historical_prices = analysis_state.historical_prices or pd.DataFrame()
                        historical_analysis = analysis_state.historical_analysis or "No analysis available."
                        news_summaries = analysis_state.news_summaries or ["No news available."]
                        recommendation_data = analysis_state.recommendation or {}
                        ticker = analysis_state.ticker

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

                        # Handle historical table data
                        historical_table_data = {'columns': [], 'data': []}
                        if not historical_prices.empty:
                            available_columns = historical_prices.columns.tolist()
                            close_col = None
                            
                            for col in available_columns:
                                if col.lower() in ['close', 'closing', 'closing_price']:
                                    close_col = col
                                    break
                            
                            if close_col:
                                columns = ['Date', close_col]
                                df_with_date = historical_prices[[close_col]].tail(5).copy()
                                df_with_date['Date'] = df_with_date.index.strftime('%Y-%m-%d') if hasattr(df_with_date.index, 'strftime') else df_with_date.index.astype(str)
                                data_records = df_with_date[['Date', close_col]].to_dict('records')
                                historical_table_data = {'columns': columns, 'data': data_records}
                            
                        report_data = {
                            "title": title,
                            "sections": [
                                {
                                    "heading": "Executive Summary",
                                    "body_text": (
                                        f"**Recommendation:** {recommendation_data.get('recommendation', 'N/A')}\n\n"
                                        f"**Rationale:** {recommendation_data.get('rationale', 'N/A')}"
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
                            "footer": "Report generated by LlamaIndex ReActAgent Agentic AI Financial Analyst"
                        }

                        filename = f"{ticker}_financial_report_llamaindex_agent.docx"
                        
                        result = await analysis_state.mcp_session.call_tool("create_professional_report", {
                            "filename": filename,
                            "report_data": report_data,
                            "table_data": historical_table_data
                        })
                        
                        return f"Report successfully generated: {filename}"
                    except Exception as e:
                        return f"Error generating report: {str(e)}"
                
                report_result = await _generate_final_report()
                logger.info(f"Report Generation Result: {report_result}")
                
                logger.info("✅ LlamaIndex ReActAgent pipeline finished.")

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}")
        raise

async def main():
    """Main function - choose approach."""
    approach = os.getenv("LLAMAINDEX_APPROACH", "workflow")  # "workflow" or "agent"
    
    if approach == "agent":
        await main_agent_approach()
    else:
        await main_workflow_approach()

if __name__ == "__main__":
    asyncio.run(main())