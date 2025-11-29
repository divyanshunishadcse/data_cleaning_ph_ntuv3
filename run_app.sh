#!/bin/bash

# Install dependencies from requirements.txt (if present)
if [ -f "requirements.txt" ]; then
	pip install -r requirements.txt
else
	pip install streamlit pandas openpyxl numpy
fi

# Allow overriding the port via env var PORT (useful on some platforms)
: ${PORT:=8501}

# Run the Streamlit app in headless mode on specified port
streamlit run app.py --server.port $PORT --server.headless true
