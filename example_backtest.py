#!/usr/bin/env python3
"""
Example script demonstrating how to use the Cryptocurrency Backtesting Framework
"""

from crypto_backtester import CryptoBacktester
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def simple_strategy_example():
    """Example 1: Simple Bitcoin trading strategy"""
    print("="*60)
    print("EXAMPLE 1: Simple Bitcoin Strategy")
    print("="*60)
    
    # Initialize backtester (API keys from environment)
    backtester = CryptoBacktester()
    
    # Define strategy in plain English
    strategy = "When Bitcoin drops 5%, buy and hold for 10 days"
    
    # Specify data sources
    excel_names = ["Bitcoin Daily Close Price"]
    
    # Parse strategy
    strategy_rules = backtester.parse_strategy_with_ai(strategy)
    
    # Fetch data
    print("\nFetching data...")
    data = backtester.fetch_data_from_pinecone(excel_names)
    
    # Run backtest
    print("\nRunning backtest...")
    backtester.run_custom_strategy(
        strategy_rules=strategy_rules,
        initial_capital=100000.0
    )
    
    # Generate reports if trades were executed
    if len(backtester.trades) > 0:
        backtester.generate_report()
        backtester.plot_equity_curve()
        backtester.print_summary()
    else:
        print("No trades executed")


def multi_asset_strategy_example():
    """Example 2: Multi-asset correlation strategy"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Asset Strategy")
    print("="*60)
    
    # Initialize backtester
    backtester = CryptoBacktester()
    
    # Define multi-asset strategy
    strategy = "When DXY rises AND 30-Year TIPS rise, buy Bitcoin 4 days later and hold for 7 days"
    
    # Specify all required data sources
    excel_names = [
        "DXY Daily Close Price",
        "30-Year TIPS Yield (%)",
        "Bitcoin Daily Close Price"
    ]
    
    # Run the complete backtest
    strategy_rules = backtester.parse_strategy_with_ai(strategy)
    data = backtester.fetch_data_from_pinecone(excel_names)
    backtester.run_custom_strategy(strategy_rules, initial_capital=250000.0)
    
    # Show results
    if len(backtester.trades) > 0:
        backtester.plot_comprehensive_dashboard()
        backtester.print_summary()


def technical_indicator_strategy():
    """Example 3: Technical indicator based strategy"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Technical Indicator Strategy")
    print("="*60)
    
    backtester = CryptoBacktester()
    
    # Moving average crossover strategy
    strategy = "Buy Bitcoin when price crosses above 20-day moving average, sell when it crosses below"
    
    excel_names = ["Bitcoin Daily Close Price"]
    
    strategy_rules = backtester.parse_strategy_with_ai(strategy)
    data = backtester.fetch_data_from_pinecone(excel_names)
    backtester.run_custom_strategy(strategy_rules, initial_capital=50000.0)
    
    if len(backtester.trades) > 0:
        backtester.plot_trade_distribution()
        print(f"\nTotal trades: {len(backtester.trades)}")
        print(f"Average trade duration: {backtester.performance_metrics.get('avg_trade_duration', 0):.1f} days")


def risk_management_strategy():
    """Example 4: Strategy with profit target and stop loss"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Risk Management Strategy")
    print("="*60)
    
    backtester = CryptoBacktester()
    
    # Strategy with risk management
    strategy = "Buy Bitcoin on 3% daily drop, exit with 5% profit or 2% loss"
    
    excel_names = ["Bitcoin Daily Close Price"]
    
    strategy_rules = backtester.parse_strategy_with_ai(strategy)
    data = backtester.fetch_data_from_pinecone(excel_names)
    backtester.run_custom_strategy(strategy_rules, initial_capital=100000.0)
    
    if len(backtester.trades) > 0:
        # Analyze risk metrics
        print(f"\nRisk Management Results:")
        print(f"Win Rate: {backtester.performance_metrics.get('percent_profitable', 0):.1f}%")
        print(f"Average Win: ${backtester.performance_metrics.get('avg_winning_trade', 0):,.2f}")
        print(f"Average Loss: ${backtester.performance_metrics.get('avg_losing_trade', 0):,.2f}")
        print(f"Risk/Reward Ratio: {backtester.performance_metrics.get('ratio_avg_win_avg_loss', 0):.2f}")


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv('PINECONE_API_KEY') or not os.getenv('OPENAI_API_KEY'):
        print("ERROR: API keys not found!")
        print("\nPlease create a .env file with:")
        print("PINECONE_API_KEY=your_key_here")
        print("OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Run examples
    try:
        # Run simple strategy
        simple_strategy_example()
        
        # Uncomment to run other examples:
        # multi_asset_strategy_example()
        # technical_indicator_strategy()
        # risk_management_strategy()
        
        print("\n✅ Examples completed successfully!")
        print("\nCheck the generated files:")
        print("- backtest_report.html")
        print("- equity_curve_interactive.html")
        print("- Various .png files for charts")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()