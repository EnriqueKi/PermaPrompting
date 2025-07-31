# Claude AI Integration Setup Guide

This guide explains how to set up the Anthropic Claude API for enhanced chromatography analysis.

## Prerequisites

1. **Get an Anthropic API Key**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Sign up or log in to your account
   - Navigate to "API Keys" section
   - Create a new API key
   - Copy the key (it starts with `sk-ant-api03-...`)

## Setup Options

### Option 1: Environment Variable (Recommended)

#### For current session only:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-actual-key-here"
```

#### For permanent setup (macOS/Linux):

Add to your shell profile (`~/.zshrc`, `~/.bashrc`, or `~/.bash_profile`):

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-your-actual-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Option 2: .env File

1. Create a `.env` file in the backend directory:

```bash
cp .env.example .env
```

2. Edit the `.env` file and add your API key:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

3. Install python-dotenv (if not already installed):

```bash
pip install python-dotenv
```

## Starting the Application with Claude Support

### Method 1: Using the startup script

```bash
# Set the API key first
export ANTHROPIC_API_KEY="your-key-here"

# Then start the application
./start.sh
```

### Method 2: Manual backend start

```bash
cd backend
export ANTHROPIC_API_KEY="your-key-here"
conda activate summerschoolenv
python api.py
```

## Verifying Claude Integration

1. **Check Claude service status:**

```bash
curl http://localhost:8080/claude/status
```

2. **Expected response with API key:**

```json
{
  "service": "Claude AI Analysis",
  "available": true,
  "api_key_configured": true,
  "endpoints": [...]
}
```

3. **Expected response without API key:**

```json
{
  "service": "Claude AI Analysis",
  "available": false,
  "api_key_configured": false,
  "endpoints": [...]
}
```

## Using Claude Analysis

1. **Upload an image** using the web interface
2. **Complete the standard analysis** first
3. **Click "Analyze with Claude AI"** button that appears after results
4. **View enhanced insights** provided by Claude

## Claude API Features

- **Image Analysis**: Detailed visual assessment of chromatograms
- **Quality Evaluation**: Assessment of separation quality and peak characteristics
- **Method Optimization**: Suggestions for improving chromatographic methods
- **Problem Identification**: Detection of issues like tailing, fronting, baseline drift
- **Intelligent Insights**: AI-powered interpretation of results

## Troubleshooting

### "Claude analysis will be disabled" message

- This means the ANTHROPIC_API_KEY is not set
- Follow the setup steps above to configure your API key

### "Claude API error" in web interface

- Check that your API key is valid and has sufficient credits
- Verify the key is properly set as an environment variable
- Check the backend logs for detailed error messages

### API key not working

- Ensure the key starts with `sk-ant-api03-`
- Check for extra spaces or characters
- Verify your Anthropic account has available credits

## Security Notes

- **Never commit your API key to version control**
- **Keep your API key secure and private**
- **Use environment variables or .env files for local development**
- **For production, use secure environment variable management**

## Cost Considerations

- Claude API usage is metered and charged per request
- Image analysis requests consume more tokens than text-only requests
- Monitor your usage in the Anthropic Console
- Consider setting usage limits if needed
