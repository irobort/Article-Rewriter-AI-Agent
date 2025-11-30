# Article Rewriter LangGraph Agent

Standalone LangGraph agent for article rewriting with browser capabilities. This production-ready agent extracts content from URLs, generates SEO-optimized articles, and humanizes the content using advanced writing style guidelines.

## Features

- **Web Content Extraction**: Automatically scrapes and extracts content from URLs
- **SEO-Optimized Article Generation**: Creates articles with target keywords, readability levels, and SEO scores
- **Content Humanization**: Applies advanced writing style guidelines for natural, engaging content
- **Structured Output**: Saves article content and metadata in separate, organized files

## Workflow

The agent follows a streamlined workflow:

1. **Extract Content** (if URL provided) - Scrapes and extracts content from the given URL
2. **Generate Article** - Creates an SEO-optimized article based on keywords, tone, and target audience
3. **Humanize Content** - Applies writing style guidelines to make content more natural and engaging

## Installation

Install dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Required Environment Variables

### Required: OpenAI API Key

Set your OpenAI API key using one of these methods:

**Option 1: Environment Variable**

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: .env File**

Create a `.env` file in the same directory as the script:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The script will automatically load the `.env` file if `python-dotenv` is installed.

### Optional: LangSmith for LLM Logging and Observability

To view detailed LLM logs, traces, and monitor agent performance, add LangSmith environment variables to your `.env` file:

```env
# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=article-rewriter-agent
```

**Getting Your LangSmith API Key:**

1. Sign up for a free account at [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to Settings → API Keys
3. Create a new API key
4. Copy the key to your `.env` file

**Benefits of LangSmith:**

- View all LLM requests and responses in real-time
- Monitor token usage and costs
- Debug agent workflow execution
- Track performance metrics
- Analyze prompt effectiveness

If LangSmith is not configured, the agent will work normally but without observability features.

## Usage

### Run as Standalone Script

```bash
python article_rewriter_agent.py
```

Or run as a module:

```bash
python -m article_rewriter_agent
```

### Customize the Example

Edit the `example_usage()` function in the script to customize:

- **URL**: The source URL to extract content from
- **Primary Keywords**: Main SEO keywords for the article
- **Secondary Keywords**: Additional keywords (optional)
- **Tone**: Writing tone (e.g., "professional", "casual", "friendly")
- **Target Word Count**: Desired article length
- **Readability Level**: "easy", "intermediate", or "advanced"
- **Target Audience**: Description of the intended readers
- **SEO Goals**: Specific SEO objectives

Example:

```python
result = await agent.process(
    url="https://example.com/article",
    primary_keywords=["keyword1", "keyword2"],
    secondary_keywords=["keyword3"],
    tone="professional",
    target_word_count=1500,
    readability_level="intermediate",
    target_audience="Marketing professionals",
    seo_goals="Rank for primary keywords",
)
```

## Output

Results are automatically saved to the `output` folder in the same directory as the script:

### File Structure

- **`{article-title}.md`** - Clean article content in markdown format (title-based filename)
- **`{article-title}_info.txt`** - Article metadata, scores, and analysis

### Output Location

- When run as a script: `{script_directory}/output/`
- When run as a module: `{current_working_directory}/output/`

The `output` folder is automatically created if it doesn't exist.

### File Contents

**Markdown File (`.md`):**
- Contains only the cleaned article content
- No title, meta description, or formatting markers
- Ready for direct use or publishing

**Info File (`.txt`):**
- Article title and word count
- Meta description and character count
- Title variations
- Readability score and suggestions
- SEO score and suggestions
- Humanization details (engagement score, changes made, improvements)

## Writing Style Guidelines

The humanization step applies strict writing style guidelines:

**SHOULD:**
- Use clear, simple language
- Be spartan and informative
- Use short, impactful sentences
- Use active voice
- Focus on practical, actionable insights
- Use "you" and "your" to directly address readers

**AVOID:**
- Em dashes (—)
- Metaphors and clichés
- Generalizations
- Setup language (in conclusion, etc.)
- Unnecessary adjectives and adverbs
- Hashtags, semicolons, markdown, asterisks
- A comprehensive list of banned words

## Dependencies

The agent requires:

- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integration
- `langgraph` - Graph-based agent orchestration
- `aiohttp` - Async HTTP client for web scraping
- `beautifulsoup4` - HTML parsing
- `python-dotenv` - Environment variable loading (optional)

All dependencies are listed in `requirements.txt`.

## Logging

The agent provides detailed logging during execution:

- Node execution status
- Content extraction progress
- Article generation parameters
- Humanization progress
- Workflow completion status

All logs are printed to the console with clear visual indicators.

## Error Handling

The agent includes comprehensive error handling:

- Invalid URL validation
- Content extraction errors
- Article generation failures
- Humanization errors
- File writing errors

Errors are logged and reported clearly, allowing for easy debugging.

# Article-Rewriter-AI-Agent
