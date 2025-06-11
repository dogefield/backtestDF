#!/usr/bin/env python3
"""
Setup script for Cryptocurrency Backtesting Framework
"""

import subprocess
import sys
import os


def check_python_version():
    """Ensure Python 3.8 or higher is installed"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"You have Python {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")


def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)


def check_env_file():
    """Check if .env file exists"""
    if os.path.exists('.env'):
        print("✅ .env file found")
        # Check if API keys are set
        with open('.env', 'r') as f:
            content = f.read()
            if 'PINECONE_API_KEY=' in content and 'OPENAI_API_KEY=' in content:
                print("✅ API keys appear to be configured")
            else:
                print("⚠️  Warning: API keys may not be properly configured in .env")
    else:
        print("\n⚠️  No .env file found!")
        print("Creating .env from .env.example...")
        try:
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created")
            print("\n⚠️  IMPORTANT: Edit .env and add your API keys!")
        except FileNotFoundError:
            print("❌ Error: .env.example not found")


def test_imports():
    """Test that all major imports work"""
    print("\nTesting imports...")
    try:
        import pandas
        print("✅ pandas imported successfully")
        
        import pinecone
        print("✅ pinecone imported successfully")
        
        import openai
        print("✅ openai imported successfully")
        
        import plotly
        print("✅ plotly imported successfully")
        
        import mplfinance
        print("✅ mplfinance imported successfully")
        
        import ta
        print("✅ ta imported successfully")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main setup function"""
    print("="*60)
    print("CRYPTOCURRENCY BACKTESTING FRAMEWORK SETUP")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check environment file
    check_env_file()
    
    # Test imports
    test_imports()
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run example: python example_backtest.py")
    print("3. Or use Jupyter: jupyter notebook EricBacktest.ipynb")
    
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()