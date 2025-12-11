# AATWW Chat Applications

This directory contains two alternative web interfaces for the AATWW (Agent and the Whole Works) multi-agent system. Choose the framework that best fits your needs.

## Directory Structure

```
apps/
├── chainlit/          # Chainlit-based chat interface
│   ├── .chainlit/
│   │   └── config.toml   # Chainlit UI configuration
│   ├── app.py            # Main Chainlit application
│   └── app.yaml          # Databricks App deployment config
├── streamlit/         # Streamlit-based chat interface
│   ├── app.py            # Main Streamlit application
│   └── app.yaml          # Databricks App deployment config
├── shared/            # Shared utilities
│   ├── __init__.py
│   └── utils.py          # Common functions for both apps
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Choosing a Framework

### Chainlit
- **Best for**: Production chat applications with a polished UI
- **Features**: Built-in conversation threading, message editing, LaTeX support, theming
- **Port**: 8000 (default)

### Streamlit
- **Best for**: Rapid prototyping and data-centric applications
- **Features**: Simple sidebar, session state management, easy to extend
- **Port**: 8501 (default)

## Running Locally

### Prerequisites
1. Set the `SERVING_ENDPOINT` environment variable:
   ```bash
   export SERVING_ENDPOINT="aatww"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Chainlit
```bash
cd apps/chainlit
chainlit run app.py --host 0.0.0.0 --port 8000
```

### Streamlit
```bash
cd apps/streamlit
streamlit run app.py
```

## Deploying to Databricks Apps

Each subdirectory contains an `app.yaml` file configured for Databricks Apps deployment.

### Deploy Chainlit version:
```bash
cd apps/chainlit
databricks apps deploy . --name aatww-chainlit
```

### Deploy Streamlit version:
```bash
cd apps/streamlit
databricks apps deploy . --name aatww-streamlit
```

## Shared Utilities

The `shared/` directory contains common functionality:

| Function | Description |
|----------|-------------|
| `get_user_info()` | Get user info from Streamlit context headers |
| `get_user_info_from_headers(headers)` | Extract user info from HTTP headers (framework-agnostic) |
| `ask_agent(input_dict, w)` | Send request using the requests library |
| `ask_agent_mlflowclient(input_dict, client)` | Send request using MLflow client |
| `extract_text_content(response_json)` | Extract text content from agent response |

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SERVING_ENDPOINT` | Name of the Databricks serving endpoint | Yes |

### Chainlit Configuration

The Chainlit UI can be customized in `chainlit/.chainlit/config.toml`:
- App name and description
- Theme colors (light/dark mode)
- Feature toggles (file upload, speech-to-text, etc.)

## Features

Both interfaces connect to the AATWW multi-agent supervisor with:
- AI/BI Genie (customer requests)
- Vector Search (product documents)
- SQL/Python functions
- External MCP (Github tools)
- Custom MCP (weather/news tools hosted on Apps)
- External API (Product recalls)
- LakeBase (memory)
- Databricks Apps
