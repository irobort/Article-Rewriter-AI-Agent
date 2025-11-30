"""
Standalone LangGraph Agent for Article Rewriting with Browser Capabilities

This is a production-ready, standalone backend agent that uses LangGraph
to orchestrate article rewriting workflows with integrated browser tools.

INSTALLATION:
    pip install -r requirements.txt

REQUIRED ENVIRONMENT VARIABLES:
    - OPENAI_API_KEY: Your OpenAI API key (required)

    You can set it in:
    1. Environment variable: export OPENAI_API_KEY=your_key
    2. .env file in the same directory as this script

USAGE:
    python article_rewriter_agent.py

    Or as a module:
    python -m article_rewriter_agent

OUTPUT:
    Results are saved to the 'output' folder:
    - article_[timestamp].md: Article content in markdown format
    - article_info_[timestamp].txt: Article metadata, scores, and analysis
"""

import logging
import json
import re
import time
import os
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from urllib.parse import urlparse
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load .env file from the same directory as this script (if running as script)
    # or from current working directory
    try:
        script_dir = Path(__file__).parent.absolute()
        env_path = script_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try current working directory
            load_dotenv()
    except NameError:
        # __file__ not available when running as module, try current directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables only

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment (will be checked when LLMService is initialized)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================================
# State Definition
# ============================================================================


class ArticleRewriterState(TypedDict):
    """State for the article rewriter agent."""

    messages: Annotated[List, lambda x, y: x + y]
    url: Optional[str]
    extracted_content: Optional[Dict[str, Any]]
    primary_keywords: List[str]
    secondary_keywords: Optional[List[str]]
    tone: str
    target_word_count: int
    readability_level: str
    target_audience: Optional[str]
    seo_goals: Optional[str]
    article_data: Optional[Dict[str, Any]]
    humanized_content: Optional[str]
    status: str
    error: Optional[str]


# ============================================================================
# Browser Tools
# ============================================================================


@tool
async def extract_url_content(url: str) -> str:
    """
    Extract content from a URL using web scraping.

    Args:
        url: The URL to extract content from

    Returns:
        JSON string containing extracted content with title, content, meta_description, headings, and word_count
    """
    try:
        logger.info(f"Extracting content from URL: {url}")

        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return json.dumps({"error": "Invalid URL format"})

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    return json.dumps(
                        {
                            "error": f"Failed to fetch URL, status code: {response.status}"
                        }
                    )

                html_content = await response.text()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "iframe", "svg"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        # Extract main content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("div", id="content")
        )

        if main_content:
            paragraphs = main_content.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
            )
        else:
            paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])

        paragraphs = [p for p in paragraphs if len(p.get_text().strip()) > 30]
        body_text = "\n\n".join([p.get_text().strip() for p in paragraphs])

        # Extract headings
        headings = {
            f"h{i}": [h.get_text().strip() for h in soup.find_all(f"h{i}")]
            for i in range(1, 7)
        }

        result = {
            "title": title,
            "content": body_text,
            "meta_description": meta_desc,
            "headings": headings,
            "word_count": len(body_text.split()),
            "url": url,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        return json.dumps({"error": f"Failed to extract content: {str(e)}"})


# ============================================================================
# LLM Service Functions
# ============================================================================


class LLMService:
    """Service for interacting with the LLM."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is not set. Please set OPENAI_API_KEY environment variable or pass it as a parameter."
            )

        self.llm = ChatOpenAI(
            # model="gpt-5-2025-08-07",
            model="gpt-4.1-2025-04-14",
            temperature=1,
            streaming=False,
            verbose=False,
            api_key=self.api_key,
        )

    async def generate_article(
        self, content: Dict[str, Any], request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate an article using the LLM."""
        try:
            logger.info("Starting article generation")

            input_content = (
                content.get("content", "")
                if content
                else request_params.get("raw_text", "")
            )
            source_title = content.get("title", "") if content else ""

            if not source_title and request_params.get("primary_keywords"):
                source_title = (
                    f"Article about {', '.join(request_params['primary_keywords'])}"
                )

            primary_keywords_str = ", ".join(request_params.get("primary_keywords", []))
            secondary_keywords_str = (
                ", ".join(request_params.get("secondary_keywords", []))
                if request_params.get("secondary_keywords")
                else ""
            )
            seo_goals = request_params.get(
                "seo_goals", f"Rank for the following keywords: {primary_keywords_str}"
            )
            target_audience = request_params.get(
                "target_audience", "General audience interested in this topic"
            )

            template = """You are an expert SEO content writer. Your task is to rewrite the provided content into an SEO-optimized article.

            Original Title: {source_title}

            Content to rewrite: {input_content}

            Requirements:
            - Target word count: {target_word_count} words
            - Tone: {tone}
            - Readability level: {readability_level}
            - Target audience: {target_audience}
            - Primary keywords: {primary_keywords}
            - Secondary keywords: {secondary_keywords}
            - SEO goals: {seo_goals}

            First, write the article content in markdown format.

            Then, after the article, provide the metadata in a JSON block like this:
            ```json
            {{
                "title": "Your generated title",
                "meta_description": {{
                    "content": "Brief description",
                    "character_count": 123
                }},
                "title_variations": [
                    {{
                        "content": "Alternative title",
                        "word_count": 3
                    }}
                ],
                "readability_score": {{
                    "score": 85,
                    "suggestions": [
                        "Suggestion 1",
                        "Suggestion 2"
                    ]
                }},
                "seo_score": {{
                    "score": 90,
                    "suggestions": [
                        "Suggestion 1",
                        "Suggestion 2"
                    ]
                }},
                "word_count": 1234
            }}
        ```"""

            from langchain.prompts import PromptTemplate
            from langchain.schema import StrOutputParser

            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "source_title",
                    "input_content",
                    "target_word_count",
                    "tone",
                    "readability_level",
                    "target_audience",
                    "primary_keywords",
                    "secondary_keywords",
                    "seo_goals",
                ],
            )

            chain = prompt | self.llm | StrOutputParser()

            output_text = await chain.ainvoke(
                {
                    "input_content": input_content,
                    "source_title": source_title,
                    "target_word_count": request_params.get("target_word_count", 1000),
                    "tone": request_params.get("tone", "professional"),
                    "readability_level": request_params.get(
                        "readability_level", "intermediate"
                    ),
                    "target_audience": target_audience,
                    "primary_keywords": primary_keywords_str,
                    "secondary_keywords": secondary_keywords_str,
                    "seo_goals": seo_goals,
                }
            )

            # Parse output
            clean_text = output_text.strip()
            content_parts = clean_text.split("```json")

            if len(content_parts) != 2:
                json_match = re.search(r"\{[\s\S]*\}", clean_text)
                if json_match:
                    json_start = json_match.start()
                    content_parts = [clean_text[:json_start], clean_text[json_start:]]
                else:
                    content_parts = [clean_text, "{}"]

            article_content = content_parts[0].strip()
            metadata_json = content_parts[1].strip()
            if metadata_json.endswith("```"):
                metadata_json = metadata_json[:-3].strip()

            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.error("Failed to parse metadata JSON, using defaults")
                metadata = {
                    "title": source_title or "Generated Article",
                    "meta_description": {
                        "content": article_content[:150] + "...",
                        "character_count": len(article_content[:150]) + 3,
                    },
                    "title_variations": [
                        {
                            "content": source_title or "Generated Article",
                            "word_count": len(
                                (source_title or "Generated Article").split()
                            ),
                        }
                    ],
                    "readability_score": {
                        "score": 70,
                        "suggestions": ["Generated due to parsing failure"],
                    },
                    "seo_score": {
                        "score": 70,
                        "suggestions": ["Generated due to parsing failure"],
                    },
                    "word_count": len(article_content.split()),
                }

            return {
                "title": metadata.get("title", source_title or "Generated Article"),
                "content": article_content,
                "meta_description": metadata.get(
                    "meta_description",
                    {
                        "content": article_content[:150] + "...",
                        "character_count": len(article_content[:150]) + 3,
                    },
                ),
                "title_variations": metadata.get(
                    "title_variations",
                    [
                        {
                            "content": metadata.get("title", "Generated Article"),
                            "word_count": len(
                                metadata.get("title", "Generated Article").split()
                            ),
                        }
                    ],
                ),
                "readability_score": metadata.get(
                    "readability_score", {"score": 70, "suggestions": []}
                ),
                "seo_score": metadata.get(
                    "seo_score", {"score": 70, "suggestions": []}
                ),
                "word_count": metadata.get("word_count", len(article_content.split())),
            }

        except Exception as e:
            logger.error(f"Error in article generation: {str(e)}")
            raise ValueError(f"Article generation failed: {str(e)}")

    async def humanize_content(
        self, content: str, tone: str, target_audience: str
    ) -> Dict[str, Any]:
        """Make the content more human-like and natural."""
        try:
            logger.info("Starting content humanization")

            from langchain.schema import StrOutputParser

            humanize_prompt = ChatPromptTemplate.from_template(
                """You are an expert in natural writing. Rewrite the following content to make it more natural and human-like while preserving its meaning and SEO elements.

                Content to humanize:
                {content}

                Requirements:
                - Maintain the same tone: {tone}
                - Target audience: {target_audience}

                WRITING STYLE GUIDELINES:

                SHOULD:
                - Use clear, simple language
                - Be spartan and informative
                - Use short, impactful sentences
                - Use active voice; avoid passive voice
                - Focus on practical, actionable insights
                - Use bullet point lists in social media posts
                - Use data and examples to support claims when possible
                - Use "you" and "your" to directly address the reader

                AVOID:
                - Em dashes (—) anywhere in your response. Use only commas, periods, or other standard punctuation. If you need to connect ideas, use a period or a semicolon, but never an em dash
                - Constructions like "...not just this, but also this"
                - Metaphors and clichés
                - Generalizations
                - Common setup language in any sentence, including: in conclusion, in closing, etc.
                - Output warnings or notes, just the output requested
                - Unnecessary adjectives and adverbs
                - Hashtags
                - Semicolons


                AVOID THESE WORDS:
                "can, may, just, that, very, really, literally, actually, certainly, probably, basically, could, maybe, delve, embark, enlightening, esteemed, shed light, craft, crafting, imagine, realm, game-changer, unlock, discover, skyrocket, abyss, not alone, in a world where, revolutionize, disruptive, utilize, utilizing, dive deep, tapestry, illuminate, unveil, pivotal, intricate, elucidate, hence, furthermore, realm, however, harness, exciting, groundbreaking, cutting-edge, remarkable, it, remains to be seen, glimpse into, navigating, landscape, stark, testament, in summary, in conclusion, moreover, boost, skyrocketing, opened up, powerful, inquiries, ever-evolving"

                IMPORTANT: Review your response and ensure no em dashes are used.


                Then, provide a JSON block:
                ```json
                {{
                    "content": "The humanized content",
                    "changes_made": ["List of major changes"],
                    "readability_improvements": ["List of improvements"],
                    "engagement_score": 85
                }}
                ```"""
            )

            chain = humanize_prompt | self.llm | StrOutputParser()

            output_text = await chain.ainvoke(
                {"content": content, "tone": tone, "target_audience": target_audience}
            )
            content_parts = output_text.split("```json")

            if len(content_parts) != 2:
                json_match = re.search(r"\{[\s\S]*\}", output_text)
                if json_match:
                    json_start = json_match.start()
                    content_parts = [output_text[:json_start], output_text[json_start:]]
                else:
                    content_parts = [output_text, "{}"]

            humanized_content = content_parts[0].strip()
            metadata_json = content_parts[1].strip()
            if metadata_json.endswith("```"):
                metadata_json = metadata_json[:-3].strip()

            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {
                    "changes_made": ["Content humanized successfully"],
                    "readability_improvements": ["Improved overall readability"],
                    "engagement_score": 80,
                }

            return {
                "content": humanized_content or content,
                "changes_made": metadata.get("changes_made", []),
                "readability_improvements": metadata.get(
                    "readability_improvements", []
                ),
                "engagement_score": metadata.get("engagement_score", 80),
            }

        except Exception as e:
            logger.error(f"Error in content humanization: {str(e)}")
            raise ValueError(f"Content humanization failed: {str(e)}")


# ============================================================================
# Graph Nodes
# ============================================================================

llm_service = LLMService()


async def extract_content_node(state: ArticleRewriterState) -> ArticleRewriterState:
    """Extract content from URL if provided."""
    try:
        if state.get("url"):
            print("\n" + "=" * 80)
            print("🔍 NODE: extract_content_node - Starting content extraction")
            print("=" * 80)
            logger.info(f"Extracting content from URL: {state['url']}")
            content_json = await extract_url_content.ainvoke(state["url"])
            content_data = json.loads(content_json)

            if "error" in content_data:
                print(f"❌ Error extracting content: {content_data['error']}")
                return {**state, "status": "error", "error": content_data["error"]}

            print(
                f"✅ Content extracted successfully (Word count: {content_data.get('word_count', 0)})"
            )
            return {
                **state,
                "extracted_content": content_data,
                "status": "content_extracted",
            }
        return state
    except Exception as e:
        print(f"❌ Error in extract_content_node: {str(e)}")
        logger.error(f"Error in extract_content_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def generate_article_node(state: ArticleRewriterState) -> ArticleRewriterState:
    """Generate the article using LLM."""
    try:
        print("\n" + "=" * 80)
        print("✍️  NODE: generate_article_node - Starting article generation")
        print("=" * 80)
        logger.info("Generating article")

        content = state.get("extracted_content") or {}
        request_params = {
            "primary_keywords": state.get("primary_keywords", []),
            "secondary_keywords": state.get("secondary_keywords"),
            "tone": state.get("tone", "professional"),
            "target_word_count": state.get("target_word_count", 1000),
            "readability_level": state.get("readability_level", "intermediate"),
            "target_audience": state.get("target_audience"),
            "seo_goals": state.get("seo_goals"),
        }

        print(
            f"📝 Parameters: {request_params.get('target_word_count')} words, {request_params.get('tone')} tone, {request_params.get('readability_level')} readability"
        )
        article_data = await llm_service.generate_article(content, request_params)
        print(
            f"✅ Article generated successfully (Word count: {article_data.get('word_count', 0)})"
        )

        return {**state, "article_data": article_data, "status": "article_generated"}
    except Exception as e:
        print(f"❌ Error in generate_article_node: {str(e)}")
        logger.error(f"Error in generate_article_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def humanize_content_node(state: ArticleRewriterState) -> ArticleRewriterState:
    """Humanize the generated content."""
    try:
        print("\n" + "=" * 80)
        print("🎨 NODE: humanize_content_node - Starting content humanization")
        print("=" * 80)
        logger.info("Humanizing content")

        article_data = state.get("article_data", {})
        content = article_data.get("content", "")

        if not content:
            return state

        humanized_result = await llm_service.humanize_content(
            content=content,
            tone=state.get("tone", "professional"),
            target_audience=state.get("target_audience", "General audience"),
        )

        # Update article data with humanized content
        article_data["content"] = humanized_result["content"]
        article_data["humanization_details"] = {
            "changes_made": humanized_result["changes_made"],
            "readability_improvements": humanized_result["readability_improvements"],
            "engagement_score": humanized_result["engagement_score"],
        }

        print(
            f"✅ Content humanized successfully (Engagement score: {humanized_result.get('engagement_score', 0)}/100)"
        )
        return {
            **state,
            "article_data": article_data,
            "humanized_content": humanized_result["content"],
            "status": "content_humanized",
        }
    except Exception as e:
        print(f"❌ Error in humanize_content_node: {str(e)}")
        logger.error(f"Error in humanize_content_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


async def agent_node(state: ArticleRewriterState) -> ArticleRewriterState:
    """Main agent node that orchestrates the workflow."""
    try:
        # Determine next step based on current status
        current_status = state.get("status", "initialized")

        if current_status == "initialized":
            # Start by extracting content if URL is provided
            if state.get("url"):
                return await extract_content_node(state)
            else:
                # Skip to article generation if no URL
                return await generate_article_node(state)

        elif current_status == "content_extracted":
            # Move to article generation
            return await generate_article_node(state)

        elif current_status == "article_generated":
            # Move to humanization
            return await humanize_content_node(state)

        elif current_status == "content_humanized":
            # Workflow complete
            return {**state, "status": "completed"}

        return state

    except Exception as e:
        logger.error(f"Error in agent_node: {str(e)}")
        return {**state, "status": "error", "error": str(e)}


def should_continue(state: ArticleRewriterState) -> Literal["continue", "end"]:
    """Determine if the workflow should continue or end."""
    status = state.get("status", "")

    if status == "completed":
        return "end"
    elif status == "error":
        return "end"
    else:
        return "continue"


# ============================================================================
# Graph Construction
# ============================================================================


def create_article_rewriter_agent():
    """Create and compile the article rewriter LangGraph agent."""

    # Create the graph
    workflow = StateGraph(ArticleRewriterState)

    # Add nodes
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )

    # Compile the graph with memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ============================================================================
# Agent Interface
# ============================================================================


class ArticleRewriterAgent:
    """Standalone LangGraph agent for article rewriting."""

    def __init__(self):
        self.graph = create_article_rewriter_agent()
        logger.info("Article Rewriter Agent initialized")

    async def process(
        self,
        url: Optional[str] = None,
        raw_text: Optional[str] = None,
        primary_keywords: List[str] = None,
        secondary_keywords: Optional[List[str]] = None,
        tone: str = "professional",
        target_word_count: int = 1000,
        readability_level: str = "intermediate",
        target_audience: Optional[str] = None,
        seo_goals: Optional[str] = None,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process an article rewriting request.

        Args:
            url: URL to extract content from (optional)
            raw_text: Raw text content to rewrite (optional)
            primary_keywords: List of primary keywords
            secondary_keywords: Optional list of secondary keywords
            tone: Tone for the article (default: professional)
            target_word_count: Target word count (default: 1000)
            readability_level: Readability level (default: intermediate)
            target_audience: Target audience description (optional)
            seo_goals: SEO goals (optional)
            thread_id: Thread ID for conversation tracking (default: default)

        Returns:
            Dictionary containing the rewritten article and metadata
        """
        try:
            # Validate inputs
            if not url and not raw_text:
                raise ValueError("Either url or raw_text must be provided")

            if not primary_keywords:
                raise ValueError("primary_keywords must be provided")

            # Prepare initial state
            initial_state = {
                "messages": [],
                "url": url,
                "extracted_content": (
                    None if url else {"content": raw_text, "title": ""}
                ),
                "primary_keywords": primary_keywords,
                "secondary_keywords": secondary_keywords,
                "tone": tone,
                "target_word_count": target_word_count,
                "readability_level": readability_level,
                "target_audience": target_audience,
                "seo_goals": seo_goals,
                "article_data": None,
                "humanized_content": None,
                "status": "initialized",
                "error": None,
            }

            # Run the agent
            print("\n" + "=" * 80)
            print("🚀 Starting Article Rewriter Agent Workflow")
            print("=" * 80)
            config = {"configurable": {"thread_id": thread_id}}
            result = None

            async for event in self.graph.astream(initial_state, config):
                result = event
                # Log progress
                if "agent" in event:
                    status = event["agent"].get("status", "processing")
                    logger.info(f"Agent status: {status}")

            print("\n" + "=" * 80)
            print("✅ Workflow completed successfully!")
            print("=" * 80)

            # Get final state
            final_state = (
                result.get("agent", initial_state) if result else initial_state
            )

            # Check for errors
            if final_state.get("status") == "error":
                error_msg = final_state.get("error", "Unknown error occurred")
                raise ValueError(f"Agent processing failed: {error_msg}")

            # Return results
            return {
                "status": "success",
                "article_data": final_state.get("article_data", {}),
                "processing_status": final_state.get("status", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error processing article rewrite request: {str(e)}")
            raise


# ============================================================================
# Factory Function
# ============================================================================


def create_agent() -> ArticleRewriterAgent:
    """Factory function to create a new article rewriter agent instance."""
    return ArticleRewriterAgent()


# ============================================================================
# Utility Functions
# ============================================================================


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """
    Sanitize a title to be used as a filename.

    Args:
        title: The title to sanitize
        max_length: Maximum length of the filename (default: 100)

    Returns:
        Sanitized filename-safe string
    """
    # Remove or replace invalid filename characters
    invalid_chars = '<>:"/\\|?*'
    sanitized = title.strip()

    for char in invalid_chars:
        sanitized = sanitized.replace(char, "-")

    # Replace multiple spaces/hyphens with single hyphen
    sanitized = re.sub(r"[\s\-]+", "-", sanitized)

    # Remove leading/trailing hyphens and dots
    sanitized = sanitized.strip(".-")

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip(".-")

    return sanitized if sanitized else "untitled"


def clean_markdown_content(content: str) -> str:
    """
    Remove markdown code block markers from content.

    Args:
        content: The content that may contain markdown code blocks

    Returns:
        Content with markdown code block markers removed
    """
    # Remove ```markdown at the start
    content = re.sub(r"^```markdown\s*\n?", "", content, flags=re.MULTILINE)
    # Remove ``` at the start (in case it's just ```)
    content = re.sub(r"^```\s*\n?", "", content, flags=re.MULTILINE)
    # Remove ``` at the end
    content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
    # Remove any remaining ``` markers
    content = re.sub(r"```", "", content)

    return content.strip()


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example of how to use the agent."""
    import os
    from datetime import datetime

    agent = create_agent()

    result = await agent.process(
        url="https://coding180.com/tutorials/learn-ai-beginners-guide/what-is-ai-basics",
        primary_keywords=["learn ai", "ai"],
        secondary_keywords=[],
        tone="professional",
        target_word_count=1000,
        readability_level="easy",
        target_audience="beginners",
        seo_goals="learn ai",
    )

    # Create output folder
    try:
        # Try to get script directory if running as script
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir / "output"
    except NameError:
        # If __file__ not available (running as module), use current working directory
        output_dir = Path.cwd() / "output"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    article_data = result.get("article_data", {})

    # Save article content to .md file
    if article_data and article_data.get("content"):
        # Get title and sanitize for filename
        title = article_data.get("title", "Untitled Article")
        sanitized_title = sanitize_filename(title)
        md_file = output_dir / f"{sanitized_title}.md"

        # Clean content to remove markdown code blocks
        content = article_data.get("content", "No content generated")
        cleaned_content = clean_markdown_content(content)

        with open(md_file, "w", encoding="utf-8") as f:
            # Write cleaned article content only (no title or meta description)
            f.write(cleaned_content)

        print(f"\n✅ Article content saved to: {md_file}")
        print(f"   Full path: {md_file.absolute()}")

    # Save other info to .txt file (without article content and RAW JSON DATA)
    # Use sanitized title for info file as well
    sanitized_title = (
        sanitize_filename(article_data.get("title", "Untitled Article"))
        if article_data
        else "article_info"
    )
    txt_file = output_dir / f"{sanitized_title}_info.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ARTICLE REWRITER AGENT - RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {result.get('status', 'unknown')}\n")
        f.write(f"Processing Status: {result.get('processing_status', 'unknown')}\n\n")

        if article_data:
            f.write("=" * 80 + "\n")
            f.write("ARTICLE DATA\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Title: {article_data.get('title', 'N/A')}\n\n")
            f.write(f"Word Count: {article_data.get('word_count', 0)}\n\n")

            if article_data.get("meta_description"):
                meta = article_data["meta_description"]
                f.write(f"Meta Description: {meta.get('content', 'N/A')}\n")
                f.write(
                    f"Meta Description Length: {meta.get('character_count', 0)} characters\n\n"
                )

            if article_data.get("title_variations"):
                f.write("-" * 80 + "\n")
                f.write("TITLE VARIATIONS\n")
                f.write("-" * 80 + "\n")
                for i, variation in enumerate(article_data["title_variations"], 1):
                    f.write(
                        f"{i}. {variation.get('content', 'N/A')} ({variation.get('word_count', 0)} words)\n"
                    )
                f.write("\n")

            if article_data.get("readability_score"):
                readability = article_data["readability_score"]
                f.write("-" * 80 + "\n")
                f.write("READABILITY SCORE\n")
                f.write("-" * 80 + "\n")
                f.write(f"Score: {readability.get('score', 0)}/100\n")
                if readability.get("suggestions"):
                    f.write("Suggestions:\n")
                    for suggestion in readability["suggestions"]:
                        f.write(f"  - {suggestion}\n")
                f.write("\n")

            if article_data.get("seo_score"):
                seo = article_data["seo_score"]
                f.write("-" * 80 + "\n")
                f.write("SEO SCORE\n")
                f.write("-" * 80 + "\n")
                f.write(f"Score: {seo.get('score', 0)}/100\n")
                if seo.get("suggestions"):
                    f.write("Suggestions:\n")
                    for suggestion in seo["suggestions"]:
                        f.write(f"  - {suggestion}\n")
                f.write("\n")

            if article_data.get("humanization_details"):
                human = article_data["humanization_details"]
                f.write("-" * 80 + "\n")
                f.write("HUMANIZATION DETAILS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Engagement Score: {human.get('engagement_score', 0)}/100\n\n")
                if human.get("changes_made"):
                    f.write("Changes Made:\n")
                    for change in human["changes_made"]:
                        f.write(f"  - {change}\n")
                    f.write("\n")
                if human.get("readability_improvements"):
                    f.write("Readability Improvements:\n")
                    for improvement in human["readability_improvements"]:
                        f.write(f"  - {improvement}\n")
                    f.write("\n")

    print(f"\n✅ Article info saved to: {txt_file}")
    print(f"   Full path: {txt_file.absolute()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
