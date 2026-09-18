# app

Local Flask server for the spider-graph frontend's `/api/generate` route.

## Run

    .venv/bin/python backend.py

Serves the frontend at http://127.0.0.1:5058/ and the API at
http://127.0.0.1:5058/api/generate.

## Stop

Ctrl+C in the terminal it's running in. If it was backgrounded:

    pkill -f "backend.py"