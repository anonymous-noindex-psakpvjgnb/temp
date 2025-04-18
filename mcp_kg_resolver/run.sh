#!/bin/bash

echo "Starting MCP Server..."
# Start server in the background
cd mcp_server
uvicorn server:app --host 0.0.0.0 --port 8100 &
SERVER_PID=$!
cd ..

echo "MCP Server started with PID $SERVER_PID. Waiting a few seconds..."
sleep 5 # Give server time to start

echo "Starting MCP Host Agent..."
cd mcp_host
python host_agent.py
cd ..

echo "Host Agent finished. Stopping MCP Server..."
kill $SERVER_PID
echo "Done."