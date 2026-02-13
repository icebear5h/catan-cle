# Build Instructions

## Prerequisites

- Python 3.11+
- `uv` package manager
- Docker (for cluster mode)
- Anthropic API key

## Setup

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Initialize project**:
   ```bash
   cd catan-learning-environment
   uv sync
   ```

3. **Set up environment**:
   ```bash
   cp .example.env .env
   # Edit .env with your API keys
   ```

4. **Initialize CLE**:
   ```bash
   cle init
   ```

## Development

TODO: Add development instructions

## Testing

```bash
uv run pytest
```

## Docker Cluster

TODO: Add Docker cluster instructions
