# Crypto Backtest Frontend

This directory contains the React dashboard that interacts with the Python backtesting API.

## Prerequisites

- Node.js 16+ and npm

## Setup

Install the dependencies:

```bash
npm install
```

(Optional) create a `.env` file to set the backend URL. The default is `http://localhost:8000`:

```bash
REACT_APP_API_URL=http://localhost:8000
```
### Using in GitHub Codespaces

When the backend runs in a Codespace, port `8000` is forwarded and a unique URL is shown in the **Ports** tab. Copy this address (for example `https://8000-<your-codespace>.githubpreview.dev`) and set `REACT_APP_API_URL` to it when starting the frontend:

```bash
REACT_APP_API_URL=https://8000-<your-codespace>.githubpreview.dev npm start
```
## Development

Start the development server:

```bash
npm start
```

## Production Build

```bash
npm run build
```

## Running with the Backend

Make sure the Python API server is running (see the main README). The dashboard will use the URL specified in `REACT_APP_API_URL` to communicate with it.

