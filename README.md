# Cryptocurrency Backtesting Framework

A sophisticated AI-powered backtesting framework for cryptocurrency trading strategies. This framework uses natural language processing to parse trading strategies and integrates with Pinecone vector database for historical data storage.

## Features

- **Natural Language Strategy Input**: Define trading strategies in plain English
- **Multi-Asset Support**: Test strategies across multiple assets simultaneously
- **Professional Visualizations**: Interactive charts using Plotly, mplfinance, and seaborn
- **Comprehensive Metrics**: Track P&L, Sharpe ratio, drawdowns, win rates, and more
- **TradingView Integration**: Import and analyze TradingView strategy tester results
- **Real-time Position Tracking**: Detailed logging of entry/exit prices and position sizes

## Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- OpenAI account and API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BacktestFile.git
cd BacktestFile
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Example

```python
from EricBacktest import CryptoBacktester

# Initialize the backtester
backtester = CryptoBacktester()

# Define strategy in natural language
strategy = "When Bitcoin drops 5%, buy and hold for 10 days"

# Specify data sources (from Pinecone)
excel_names = ["Bitcoin Daily Close Price"]

# Parse strategy
strategy_rules = backtester.parse_strategy_with_ai(strategy)

# Fetch data
data = backtester.fetch_data_from_pinecone(excel_names)

# Run backtest
backtester.run_custom_strategy(
    strategy_rules=strategy_rules,
    initial_capital=100000.0
)

# Generate reports and visualizations
backtester.generate_report()
backtester.plot_equity_curve()
backtester.print_summary()
```

### Multi-Asset Strategy Example

```python
# Define a multi-asset strategy
strategy = "When DXY rises AND 30-Year TIPS rise, buy Bitcoin 4 days later"
excel_names = ["DXY Daily Close Price", "30-Year TIPS Yield (%)", "Bitcoin Daily Close Price"]

# Run the backtest
backtester = CryptoBacktester()
strategy_rules = backtester.parse_strategy_with_ai(strategy)
data = backtester.fetch_data_from_pinecone(excel_names)
backtester.run_custom_strategy(strategy_rules, initial_capital=100000.0)
```

### Strategy Examples

The framework understands various natural language patterns:

- **Simple Buy/Sell**: "Buy Bitcoin when it drops 10% and sell after 5 days"
- **Technical Indicators**: "Buy ETH when price crosses above 20-day moving average"
- **Multi-Asset Conditions**: "When SPY drops AND VIX rises, buy Bitcoin 3 days later"
- **Profit/Loss Targets**: "Buy BTC on 5% drop, sell at 10% profit or 5% loss"

## Output Files

The framework generates several output files:

- `backtest_report.html` - Comprehensive performance report
- `equity_curve_interactive.html` - Interactive P&L visualization
- `trade_distribution_interactive.html` - Trade analysis dashboard
- `comprehensive_dashboard.png` - Summary visualization
- `bitcoin_price_chart.png` - Price chart with trade markers

## Performance Metrics

The framework tracks extensive metrics including:

- **P&L Metrics**: Net profit, gross profit/loss, commission costs
- **Trade Statistics**: Win rate, average win/loss, largest trades
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Position Metrics**: Bars in trade, run-up/drawdown per trade
- **Long/Short Analysis**: Separate metrics for long and short positions

## Data Structure

The framework expects data in Pinecone with the following metadata structure:
- `excel_name`: Identifier for the data series
- `raw_text`: Contains date and price information
- `Date`: Trading date
- `Close`: Closing price

## Customization

### Strategy Properties

You can customize various parameters:

```python
backtester.strategy_properties = {
    'position_size_percent': 100,  # Use 100% of capital
    'holding_period': 10,          # Days to hold positions
    'commission_rate': 0.001,      # 0.1% commission
    'slippage': 0,                 # No slippage
    'margin_long': 0,              # No margin for longs
    'margin_short': 0              # No margin for shorts
}
```

### Custom Indicators

Add custom technical indicators in the `add_basic_indicators()` method.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file contains valid API keys
2. **Data Not Found**: Verify excel names match exactly with Pinecone data
3. **No Trades Executed**: Check strategy conditions and data availability
4. **Visualization Errors**: Install all required libraries from requirements.txt

### Debug Mode

Enable detailed logging by checking intermediate outputs:

```python
# Check parsed strategy
print(json.dumps(strategy_rules, indent=2))

# Check data loading
print(f"Data shape: {backtester.data.shape}")
print(backtester.data.head())

# Check trade execution
for trade in backtester.trades[:5]:
    print(trade)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Pinecone vector database
- Powered by OpenAI GPT-4 for strategy parsing
- Visualization libraries: Plotly, mplfinance, seaborn
- Technical analysis: ta library