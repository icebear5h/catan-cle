"""Probe the Colonist API endpoints with an optional JWT token.

Usage: python -m data_pipeline.bootstrapping.scrapers.colonist_api <jwt_token>
       COLONIST_JWT=<token> python -m data_pipeline.bootstrapping.scrapers.colonist_api
"""

import asyncio
import os
import sys

from data_pipeline.bootstrapping.scrapers.colonist_api import discover_api_endpoints

# Get JWT from environment or command line
jwt_token = os.environ.get("COLONIST_JWT")
if len(sys.argv) > 1:
    jwt_token = sys.argv[1]

if jwt_token:
    print(f"Using JWT token: {jwt_token[:20]}...")
else:
    print("No JWT token provided. Set COLONIST_JWT env var or pass as argument.")
    print("Usage: python colonist_api.py <jwt_token>")
    print("       COLONIST_JWT=<token> python colonist_api.py")

# Run endpoint discovery
asyncio.run(discover_api_endpoints(jwt_token))
