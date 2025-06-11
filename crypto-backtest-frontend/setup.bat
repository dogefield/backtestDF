@echo off
echo 🚀 Setting up Crypto Backtest Frontend...

echo 📦 Installing dependencies...
npm install

echo 🔧 Creating environment file...
copy .env.example .env

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run 'npm start' to start the development server
echo 3. Open http://localhost:3000 in your browser
