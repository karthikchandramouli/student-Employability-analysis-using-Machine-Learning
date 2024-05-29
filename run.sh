#!/bin/bash

# Activate virtual environment if necessary
# source venv/bin/activate

# Install required packages
#pip install -r requirements.txt

# Run Flask application
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
