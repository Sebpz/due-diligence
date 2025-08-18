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
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import threading
from concurrent.futures import ThreadPoolExecutor

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
        self._loop: Optional[asyncio.AbstractEventLoop] = None

# Global state instance
analysis_state = None

# Helper function for OpenAI API calls with exponential backoff
async def call_openai_api_with_exponential_backoff(payload, max_retries=3):
    """Make OpenAI API call with exponential backoff."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_endpoint = os.getenv("OPENAI_ENDPOINT")
    
    if not openai_api_key or not openai_endpoint:
        logger.error("OPENAI_API_KEY and OPENAI_ENDPOINT must be set")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    
    logger.info(f"Making API call to: {openai_endpoint}")
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(openai_endpoint, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
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

# Helper function to run coroutines in the background event loop
def run_async_in_thread(coro):
    """Run async function in the background thread's event loop."""
    if analysis_state._loop is None:
        raise RuntimeError("Background event loop not initialized")
    
    future = asyncio.run_coroutine_threadsafe(coro, analysis_state._loop)
    return future.result()

# CrewAI Tools with fixed async handling
class CompanyDataInput(BaseModel):
    ticker_symbol: str = Field(description="Stock ticker symbol to fetch data for")

class CompanyDataTool(BaseTool):
    name: str = "get_company_data"
    description: str = "Fetch company financial data and stock prices"
    args_schema: type[BaseModel] = CompanyDataInput

    def _run(self, ticker_symbol: str) -> str:
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
                        
                        return f"Successfully fetched data for {ticker_symbol}. Company: {analysis_state.company_info.get('shortName', 'N/A')}. Analysis: {analysis_state.historical_analysis}"
                    else:
                        return f"Error fetching data: {data.get('error')}"
                return "No data received from MCP tool"
            except Exception as e:
                return f"Error fetching company data: {str(e)}"
        
        try:
            return run_async_in_thread(_fetch_data())
        except Exception as e:
            return f"Error running async function: {str(e)}"

class NewsSearchInput(BaseModel):
    company_name: str = Field(description="Company name to search news for")

class NewsSearchTool(BaseTool):
    name: str = "search_news"
    description: str = "Search and summarize financial news for a company"
    args_schema: type[BaseModel] = NewsSearchInput

    def _run(self, company_name: str) -> str:
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
                return f"Found and summarized {len(summaries)} news articles: {'; '.join(summaries[:2])}"
            except Exception as e:
                analysis_state.news_summaries = ["Error during news retrieval and summarization."]
                return f"Error searching news: {str(e)}"
        
        try:
            return run_async_in_thread(_fetch_news())
        except Exception as e:
            return f"Error running async function: {str(e)}"

class InvestmentRecommendationInput(BaseModel):
    analysis_data: str = Field(description="Combined financial and news analysis data")

class InvestmentRecommendationTool(BaseTool):
    name: str = "generate_investment_recommendation"
    description: str = "Generate investment recommendation based on financial analysis and news"
    args_schema: type[BaseModel] = InvestmentRecommendationInput

    def _run(self, analysis_data: str) -> str:
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
                return f"Investment Recommendation: {recommendation_data.get('recommendation')} - {recommendation_data.get('rationale')}"
            except Exception as e:
                analysis_state.recommendation = {"recommendation": "Hold", "rationale": "Error during generation."}
                return f"Error generating recommendation: {str(e)}"
        
        try:
            return run_async_in_thread(_generate_recommendation())
        except Exception as e:
            return f"Error running async function: {str(e)}"

class ReportGenerationInput(BaseModel):
    report_request: str = Field(description="Request to generate the final report")

class ReportGenerationTool(BaseTool):
    name: str = "generate_financial_report"
    description: str = "Generate a comprehensive financial report document"
    args_schema: type[BaseModel] = ReportGenerationInput

    def _run(self, report_request: str) -> str:
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
                    "footer": "Report generated by CrewAI Agentic AI Financial Analyst (Fixed Version)"
                }

                filename = f"{ticker}_financial_report_crewai_fixed.docx"
                
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
                    return f"Report successfully generated: {filename}"
                else:
                    return f"Report generation failed: {response_data.get('message', 'Unknown error')}"
            except Exception as e:
                return f"Error generating report: {str(e)}"
        
        try:
            return run_async_in_thread(_generate_report())
        except Exception as e:
            return f"Error running async function: {str(e)}"

# Background event loop runner
def run_background_loop(loop):
    """Run the event loop in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def initialize_mcp_session():
    """Initialize MCP session in the background loop."""
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            analysis_state.mcp_session = session
            
            # Keep the session alive
            while True:
                await asyncio.sleep(1)

def main():
    global analysis_state
    
    try:
        # Initialize shared state
        analysis_state = FinancialAnalysisState("GOOG")
        
        # Create background event loop
        background_loop = asyncio.new_event_loop()
        analysis_state._loop = background_loop
        
        # Start background thread for async operations
        background_thread = threading.Thread(
            target=run_background_loop,
            args=(background_loop,),
            daemon=True
        )
        background_thread.start()
        
        # Initialize MCP session in background loop
        future = asyncio.run_coroutine_threadsafe(
            initialize_mcp_session(),
            background_loop
        )
        
        # Give some time for initialization
        import time
        time.sleep(2)
        
        # Initialize tools
        company_data_tool = CompanyDataTool()
        news_search_tool = NewsSearchTool()
        recommendation_tool = InvestmentRecommendationTool()
        report_tool = ReportGenerationTool()

        # Configure LLM for CrewAI agents
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        
        if not openai_api_key or not openai_endpoint:
            raise ValueError("OPENAI_API_KEY and OPENAI_ENDPOINT must be set in environment variables")
        
        # Extract deployment name from endpoint for Azure OpenAI
        import re
        deployment_match = re.search(r'/deployments/([^/]+)/', openai_endpoint)
        deployment_name = deployment_match.group(1) if deployment_match else "gpt-4o"
        
        # Extract base URL
        base_url_match = re.search(r'(https://[^/]+)', openai_endpoint)
        base_url = base_url_match.group(1) if base_url_match else openai_endpoint
        
        # Configure LLM for CrewAI
        llm = LLM(
            model=f"azure/{deployment_name}",
            api_key=openai_api_key,
            base_url=base_url,
            api_version="2025-01-01-preview"
        )

        # Define CrewAI Agents with LLM configuration
        data_analyst = Agent(
            role='Financial Data Analyst',
            goal='Fetch and analyze company financial data and stock performance',
            backstory="""You are an expert financial data analyst with deep knowledge of 
            market analysis and company fundamentals. You specialize in extracting insights 
            from financial data and stock price movements.""",
            verbose=True,
            allow_delegation=False,
            tools=[company_data_tool],
            llm=llm
        )

        news_analyst = Agent(
            role='Financial News Analyst',
            goal='Search, collect, and summarize relevant financial news',
            backstory="""You are a seasoned financial journalist and news analyst. 
            You have a keen eye for identifying market-moving news and can quickly 
            summarize complex financial information for investment decision making.""",
            verbose=True,
            allow_delegation=False,
            tools=[news_search_tool],
            llm=llm
        )

        investment_advisor = Agent(
            role='Senior Portfolio Manager',
            goal='Provide actionable investment recommendations based on comprehensive analysis',
            backstory="""You are a senior portfolio manager with 20+ years of experience 
            in equity analysis and investment decision making. You excel at synthesizing 
            complex financial data and market sentiment into clear, actionable investment recommendations.""",
            verbose=True,
            allow_delegation=False,
            tools=[recommendation_tool],
            llm=llm
        )

        report_generator = Agent(
            role='Financial Report Writer',
            goal='Create comprehensive, professional financial analysis reports',
            backstory="""You are an expert financial report writer who specializes in 
            creating clear, comprehensive investment reports that combine technical analysis, 
            fundamental analysis, and market sentiment into actionable insights.""",
            verbose=True,
            allow_delegation=False,
            tools=[report_tool],
            llm=llm
        )

        # Define Tasks
        data_analysis_task = Task(
            description=f"""Fetch and analyze financial data for {analysis_state.ticker}. 
            Extract key financial metrics, current stock price, market cap, P/E ratio, 
            and perform historical price analysis including moving averages and trends.""",
            expected_output="A comprehensive financial analysis with key metrics and price trend analysis",
            agent=data_analyst
        )

        news_analysis_task = Task(
            description=f"""Search for recent financial news about the company and 
            summarize the most relevant articles. Focus on market sentiment, company 
            performance, and any factors that could impact stock price.""",
            expected_output="A summary of recent relevant financial news with key insights",
            agent=news_analyst
        )

        recommendation_task = Task(
            description="""Based on the financial analysis and news summaries, 
            generate a clear investment recommendation (Buy, Hold, or Sell) with 
            detailed rationale citing the supporting evidence.""",
            expected_output="A clear investment recommendation with detailed justification",
            agent=investment_advisor
        )

        report_generation_task = Task(
            description="""Create a comprehensive financial report that includes 
            executive summary with recommendation, company overview, financial metrics, 
            historical analysis, and news summary. Generate a professional Word document.""",
            expected_output="A complete financial analysis report saved as a Word document",
            agent=report_generator
        )

        # Create and run the crew
        crew = Crew(
            agents=[data_analyst, news_analyst, investment_advisor, report_generator],
            tasks=[data_analysis_task, news_analysis_task, recommendation_task, report_generation_task],
            verbose=True,
            process=Process.sequential
        )

        logger.info("🚀 Starting Fixed CrewAI multi-agent financial analysis pipeline...")
        
        result = crew.kickoff()
        
        logger.info("✅ Fixed CrewAI pipeline finished. Report has been generated.")
        logger.info(f"Final Result: {result}")
        
        # Cleanup
        background_loop.call_soon_threadsafe(background_loop.stop)

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}")
        raise
    finally:
        # Cleanup background loop
        if 'background_loop' in locals():
            background_loop.call_soon_threadsafe(background_loop.stop)

if __name__ == "__main__":
    main()