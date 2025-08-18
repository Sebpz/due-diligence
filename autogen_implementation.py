import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv
import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent

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
    
    # Determine if using Azure OpenAI or regular OpenAI
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

# MCP Tool Functions
async def get_company_data(ticker_symbol: str) -> dict:
    """Fetch company data using MCP tool."""
    try:
        result = await analysis_state.mcp_session.call_tool("get_company_data", {"ticker_symbol": ticker_symbol})
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content[0], dict):
                return result.content[0]
            else:
                return json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
        return {"error": "No data received"}
    except Exception as e:
        logger.error(f"Error calling get_company_data: {e}")
        return {"error": str(e)}

async def search_for_news(query: str) -> list:
    """Search for news using MCP tool."""
    try:
        result = await analysis_state.mcp_session.call_tool("search_for_news", {"query": query})
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content[0], dict):
                return result.content[0] if isinstance(result.content[0], list) else []
            else:
                return json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else []
        return []
    except Exception as e:
        logger.error(f"Error calling search_for_news: {e}")
        return []

async def get_soup_from_url(url: str) -> dict:
    """Get webpage content using MCP tool."""
    try:
        result = await analysis_state.mcp_session.call_tool("get_soup_from_url", {"url": url})
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content[0], dict):
                return result.content[0]
            else:
                return json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
        return {"error": "No data received"}
    except Exception as e:
        logger.error(f"Error calling get_soup_from_url: {e}")
        return {"error": str(e)}

async def create_professional_report(filename: str, report_data: dict, table_data: Optional[dict] = None) -> dict:
    """Create professional report using MCP tool."""
    try:
        result = await analysis_state.mcp_session.call_tool("create_professional_report", {
            "filename": filename,
            "report_data": report_data,
            "table_data": table_data
        })
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content[0], dict):
                return result.content[0]
            else:
                return json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else {}
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error calling create_professional_report: {e}")
        return {"status": "error", "message": str(e)}

async def data_analysis_agent_function(message: str) -> str:
    """Data Analysis Agent function."""
    logger.info(f"📊 Data Analysis Agent: Fetching and analyzing data for {analysis_state.ticker}...")
    
    async def _fetch_data():
        try:
            # Initialize state with empty values
            analysis_state.company_info = {}
            analysis_state.historical_prices = pd.DataFrame()
            analysis_state.historical_analysis = "No analysis available"

            data = await get_company_data(analysis_state.ticker)
            
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
                
                return f"✅ Successfully fetched data for {analysis_state.ticker}. Company: {analysis_state.company_info.get('shortName', 'N/A')}. Analysis: {analysis_state.historical_analysis}"
            else:
                return f"❌ Error fetching data: {data.get('error')}"
        except Exception as e:
            return f"❌ Error in data analysis: {str(e)}"
    
    # Run the async function

    return await _fetch_data()

async def news_summarization_agent_function(message: str) -> str:
    """News Summarization Agent function."""
    logger.info(f"📰 News Summarization Agent: Searching for news about {analysis_state.ticker}")
    
    async def _fetch_and_summarize_news():
        try:
            company_name = analysis_state.company_info.get('shortName', analysis_state.ticker)
            analysis_state.news_summaries = []

            news_articles = await search_for_news(company_name)
            
            if not news_articles:
                analysis_state.news_summaries = ["No recent news found."]
                return "No news articles found."

            summaries = []
            for article in news_articles[:3]:
                url = article.get('url', 'N/A')
                try:
                    page_content = await get_soup_from_url(url)
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
            return f"✅ Found and summarized {len(summaries)} news articles."
        except Exception as e:
            analysis_state.news_summaries = ["Error during news retrieval and summarization."]
            return f"❌ Error in news summarization: {str(e)}"
    
    
    return await _fetch_and_summarize_news()

async def recommendation_agent_function(message: str) -> str:
    """Recommendation Agent function."""
    logger.info("💡 Recommendation Agent: Synthesizing data for an investment recommendation...")
    
    async def _generate_recommendation():
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
            return f"✅ Recommendation: {recommendation_data.get('recommendation')} - {recommendation_data.get('rationale')}"
        except Exception as e:
            analysis_state.recommendation = {"recommendation": "Hold", "rationale": "Error during generation."}
            return f"❌ Error generating recommendation: {str(e)}"
    
    return await _generate_recommendation()

async def report_generation_agent_function(message: str) -> str:
    """Report Generation Agent function."""
    logger.info("✍️ Report Generation Agent: Assembling the final report...")
    
    async def _generate_report():
        try:
            company_info = analysis_state.company_info or {}
            historical_prices = analysis_state.historical_prices
            if historical_prices is None:
                historical_prices = pd.DataFrame()
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
                "footer": "Report generated by AutoGen Agentic AI Financial Analyst"
            }

            filename = f"{ticker}_financial_report_autogen.docx"
            result = await create_professional_report(filename, report_data, historical_table_data)
            
            if result.get('status') == 'success':
                analysis_state.final_report = report_data
                return f"✅ Report successfully generated: {filename}"
            else:
                return f"❌ Report generation failed: {result.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error generating report: {str(e)}"
    
    return await _generate_report()

async def main():
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
                
                # Configure AutoGen agents
                config_list = [{
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": f"{os.getenv('OPENAI_ENDPOINT')}", #/openai/deployments/gpt-4o" if os.getenv('OPENAI_ENDPOINT') else "https://api.openai.com/v1",
                    # "api_type": "azure" if os.getenv('OPENAI_ENDPOINT') else "openai",
                    # "api_version": "2025-01-01-preview"  #if os.getenv('OPENAI_ENDPOINT') else None
                }]
                
                # Create AutoGen agents
                user_proxy = UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=10,
                    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                    code_execution_config={"work_dir": "workspace"},
                )

                # Data Analysis Agent
                data_agent = ConversableAgent(
                    name="data_analyst",
                    system_message="You are a financial data analyst. Your role is to fetch and analyze company financial data.",
                    llm_config={"config_list": config_list},
                    function_map={"data_analysis": data_analysis_agent_function}
                )

                # News Summarization Agent
                news_agent = ConversableAgent(
                    name="news_analyst",
                    system_message="You are a news analyst. Your role is to find and summarize relevant financial news.",
                    llm_config={"config_list": config_list},
                    function_map={"news_analysis": news_summarization_agent_function}
                )

                # Recommendation Agent
                recommendation_agent = ConversableAgent(
                    name="investment_advisor",
                    system_message="You are a senior portfolio manager. Your role is to provide investment recommendations.",
                    llm_config={"config_list": config_list},
                    function_map={"generate_recommendation": recommendation_agent_function}
                )

                # Report Generation Agent
                report_agent = ConversableAgent(
                    name="report_generator",
                    system_message="You are a report generator. Your role is to create comprehensive financial reports.",
                    llm_config={"config_list": config_list},
                    function_map={"generate_report": report_generation_agent_function}
                )

                logger.info("🚀 Starting AutoGen multi-agent financial analysis pipeline...")

                # Execute the pipeline sequentially
                logger.info("Step 1: Data Analysis")
                data_result = await data_analysis_agent_function("Analyze financial data")
                logger.info(f"Data Analysis Result: {data_result}")

                logger.info("Step 2: News Analysis")
                news_result = await news_summarization_agent_function("Summarize news")
                logger.info(f"News Analysis Result: {news_result}")

                logger.info("Step 3: Generate Recommendation")
                rec_result = await recommendation_agent_function("Generate recommendation")
                logger.info(f"Recommendation Result: {rec_result}")

                logger.info("Step 4: Generate Report")
                report_result = await report_generation_agent_function("Generate report")
                logger.info(f"Report Generation Result: {report_result}")

                logger.info("✅ AutoGen pipeline finished. Report has been generated.")

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())