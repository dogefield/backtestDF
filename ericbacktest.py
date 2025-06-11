import pandas as pd
import numpy as np
from pinecone import Pinecone
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import json
import warnings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

# Professional trading visualization libraries
try:
    import mplfinance as mpf
except ImportError:
    mpf = None
    print("Warning: mplfinance not installed; certain charts will be disabled.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    go = make_subplots = px = None
    print("Warning: plotly not installed; interactive charts disabled.")

try:
    import ta  # Technical Analysis library
except ImportError:
    ta = None
    print("Warning: ta library not installed; indicators may be unavailable.")

class CryptoBacktester:
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None):
        """Initialize the backtesting framework with Pinecone and OpenAI connections"""
        # Use environment variables if API keys not provided
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Validate API keys
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable or pass it as parameter.")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it as parameter.")
        
        self.index_name = "intelligence-main"
        self.data = None
        self.trades = []
        self.performance_metrics = {}
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize OpenAI with new client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Strategy properties storage
        self.strategy_properties = {
            'position_size_percent': 100,
            'holding_period': None,
            'commission_rate': 0.001,
            'slippage': 0,
            'margin_long': 0,
            'margin_short': 0
        }
        
        # Track max positions
        self.max_contracts_held = 0
        self.max_contracts_held_long = 0
        self.max_contracts_held_short = 0
        
        # Professional color scheme
        self.colors = {
            'green': '#00D775',  # Professional trading green
            'red': '#FF3366',    # Professional trading red
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'background': '#0e1117',
            'grid': '#262730'
        }
        
    def parse_strategy_with_ai(self, strategy_description: str) -> Dict:
        """Use OpenAI to parse natural language strategy description into trading rules"""
        
        prompt = f"""
        Parse the following trading strategy description into structured trading rules.
        
        Strategy: "{strategy_description}"
        
        Return a JSON object with the following structure:
        {{
            "entry_conditions": {{
                "primary_asset": "string (e.g., BTC, Bitcoin, SPY)",
                "trigger_asset": "string (asset to monitor for signals, can be same as primary)",
                "condition": "string (e.g., 'drops', 'rises', 'crosses_above', 'crosses_below')",
                "threshold": "number or null (percentage or absolute value)",
                "delay_days": "number (how many days to wait after trigger)",
                "additional_filters": []
            }},
            "exit_conditions": {{
                "type": "string (e.g., 'time_based', 'profit_target', 'stop_loss', 'condition_based')",
                "value": "number (days for time_based, percentage for profit/loss)",
                "condition": "string (optional, for condition-based exits)"
            }},
            "position_size": "number (fraction of capital to use, default 0.1)",
            "position_type": "string (long or short)"
        }}
        
        Examples:
        - "when SPY goes down buy BTC 5 days later" -> trigger on SPY drop, buy BTC after 5 days
        - "buy ETH when it drops 10% and sell after 7 days" -> 10% drop trigger, 7 day holding period
        - "When DXY up AND 30-Year TIPS up, BUY Bitcoin 4 days later" -> Both DXY and TIPS must rise, then buy Bitcoin
        
        For multi-asset conditions with AND, list all trigger assets separated by commas.
        If exit conditions are not specified, use a default 10 day holding period.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading strategy parser. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            strategy_rules = json.loads(response.choices[0].message.content)
            
            # Add defaults if missing
            if 'position_size' not in strategy_rules:
                strategy_rules['position_size'] = 0.1
            if 'position_type' not in strategy_rules:
                strategy_rules['position_type'] = 'long'
            
            # Store strategy properties
            self.strategy_properties['position_size_percent'] = strategy_rules['position_size'] * 100
            if strategy_rules['exit_conditions']['type'] == 'time_based':
                self.strategy_properties['holding_period'] = strategy_rules['exit_conditions']['value']
            
            print("\nParsed Strategy Rules:")
            print(json.dumps(strategy_rules, indent=2))
            
            return strategy_rules
            
        except Exception as e:
            print(f"Error parsing strategy: {e}")
            # Return default rules
            return {
                "entry_conditions": {
                    "primary_asset": "BTC",
                    "trigger_asset": "BTC",
                    "condition": "drops",
                    "threshold": 5,
                    "delay_days": 0,
                    "additional_filters": []
                },
                "exit_conditions": {
                    "type": "time_based",
                    "value": 10
                },
                "position_size": 0.1,
                "position_type": "long"
            }
    
    def _parse_raw_text(self, raw_text: str) -> Dict:
        """Parse the raw_text field to extract key-value pairs"""
        data = {}
        parts = raw_text.split(' | ')
        
        for part in parts:
            if ': ' in part:
                try:
                    key, value = part.split(': ', 1)
                    # Clean up keys - handle various formats
                    if 'Date' in key:
                        data['Date'] = value
                    elif 'Symbol' in key:
                        data['Symbol'] = value
                    else:
                        # Try to extract numeric value from any field
                        # This handles Close Price, Daily Close Price, DXY, Bitcoin, TIPS Yield, etc.
                        clean_value = value.replace('$', '').replace(',', '').replace('%', '').strip()
                        try:
                            numeric_value = float(clean_value)
                            # If we successfully parsed a number and don't have a Close value yet
                            if 'Close' not in data and numeric_value != 0:
                                data['Close'] = numeric_value
                                # Track if this is a yield value
                                if 'Yield' in key:
                                    data['is_yield'] = True
                        except ValueError:
                            # Not a numeric value, skip
                            continue
                except ValueError as e:
                    continue
        
        # If no symbol found, try to infer from the data
        if 'Symbol' not in data:
            # Look for common symbols in the raw text
            if 'BTC' in raw_text or 'Bitcoin' in raw_text:
                data['Symbol'] = 'BTC'
            elif 'DXY' in raw_text:
                data['Symbol'] = 'DXY'
            elif 'TIPS' in raw_text:
                data['Symbol'] = 'TIPS'
            else:
                data['Symbol'] = 'UNKNOWN'
        
        return data
    
    def fetch_data_from_pinecone(self, excel_names: List[str]) -> pd.DataFrame:
        """Fetch all data for specified excel_names from Pinecone"""
        all_data = []
        
        for i, excel_name in enumerate(excel_names):
            print(f"\nFetching data for: {excel_name}")
            # Query Pinecone for vectors with matching excel_name
            query_response = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata filtering
                filter={"excel_name": {"$eq": excel_name}},
                top_k=10000,  # Get all matching records
                include_metadata=True
            )
            
            print(f"Found {len(query_response['matches'])} records")
            
            # Show sample raw_text for debugging
            if query_response['matches'] and i == 0:  # Only show for first excel_name
                sample_metadata = query_response['matches'][0]['metadata']
                sample_raw_text = sample_metadata.get('raw_text', '')
                print(f"\nSample raw_text format:")
                print(f"{sample_raw_text[:300]}...")
                
                # Parse the sample to show what we're extracting
                sample_parsed = self._parse_raw_text(sample_raw_text)
                print(f"\nParsed fields: {sample_parsed}")
            
            # Extract data from matches
            for match in query_response['matches']:
                metadata = match['metadata']
                # Parse the raw_text to extract values
                raw_text = metadata.get('raw_text', '')
                data_dict = self._parse_raw_text(raw_text)
                
                # Skip if we couldn't parse essential fields
                if 'Close' not in data_dict or 'Date' not in data_dict:
                    if i == 0 and match == query_response['matches'][0]:  # Only show once
                        print(f"\nSkipping record - missing Close or Date:")
                        print(f"Raw text: {raw_text[:100]}...")
                        print(f"Parsed: {data_dict}")
                    continue
                    
                data_dict.update({
                    'id': match['id'],
                    'excel_name': metadata.get('excel_name'),
                    'filename': metadata.get('filename'),
                    'sheet': metadata.get('sheet'),
                    'row_idx': metadata.get('row_idx')
                })
                all_data.append(data_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        if len(df) == 0:
            raise ValueError(f"No data found for excel names: {excel_names}")
            
        # Convert Date column with error handling
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Remove any rows with invalid dates
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
        except Exception as e:
            print(f"Error parsing dates: {e}")
            print(f"Sample Date values: {df['Date'].head()}")
            raise
        
        # Print available columns for debugging
        print(f"\nTotal records loaded: {len(df)}")
        print(f"Available columns: {df.columns.tolist()}")
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nSample data (first 3 rows):")
        print(df.head(3))
        
        # Convert Close column to numeric
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        else:
            raise ValueError("No 'Close' price column found in data")
        
        self.data = df
        return df
    
    def add_basic_indicators(self):
        """Add basic technical indicators needed for strategy execution"""
        if self.data is None:
            raise ValueError("No data loaded. Please fetch data first.")
        
        # For close-only data, use Close price directly
        if 'Close' not in self.data.columns:
            print(f"Error: No 'Close' price column found.")
            print(f"Available columns: {self.data.columns.tolist()}")
            print(f"Sample data:\n{self.data.head()}")
            
            # Try to find any price-like column
            price_columns = ['Price', 'price', 'Value', 'value']
            for col in price_columns:
                if col in self.data.columns:
                    print(f"Using '{col}' column as price data")
                    self.data['Close'] = self.data[col]
                    break
            
            if 'Close' not in self.data.columns:
                raise ValueError(f"No 'Close' price column found. Available columns: {self.data.columns.tolist()}")
        
        # Use Close price for all calculations
        self.data['Price'] = self.data['Close']
        
        # Calculate returns and percentage changes
        self.data['Return'] = self.data['Price'].pct_change()
        self.data['Cumulative_Return'] = (1 + self.data['Return']).cumprod()
        
        # Rolling statistics
        self.data['SMA_5'] = self.data['Price'].rolling(window=5).mean()
        self.data['SMA_20'] = self.data['Price'].rolling(window=20).mean()
        
        # Price changes over different periods
        for days in [1, 3, 5, 7, 10]:
            self.data[f'Price_Change_{days}d'] = self.data['Price'].pct_change(days) * 100
            
        print(f"\nIndicators added. Data shape: {self.data.shape}")
    
    def run_custom_strategy(self, strategy_rules: Dict, initial_capital: float = 100000.0):
        """Run a custom strategy based on parsed rules"""
        if self.data is None:
            raise ValueError("No data loaded. Please fetch data first.")
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        
        # Add basic indicators
        self.add_basic_indicators()
        
        # For multi-asset strategies, we need to organize data by asset
        # Group data by excel_name to handle multiple assets
        asset_data = {}
        for excel_name in self.data['excel_name'].unique():
            asset_df = self.data[self.data['excel_name'] == excel_name].copy()
            asset_data[excel_name] = asset_df
            print(f"\nAsset: {excel_name}")
            print(f"  Records: {len(asset_df)}")
            print(f"  Date range: {asset_df['Date'].min()} to {asset_df['Date'].max()}")
            
        # Check for date alignment issues
        all_dates = set()
        for excel_name, df in asset_data.items():
            all_dates.update(df['Date'].unique())
        
        print(f"\nTotal unique dates across all assets: {len(all_dates)}")
        
        # Find common dates across all assets
        common_dates = None
        for excel_name, df in asset_data.items():
            asset_dates = set(df['Date'].unique())
            if common_dates is None:
                common_dates = asset_dates
            else:
                common_dates = common_dates.intersection(asset_dates)
        
        print(f"Common dates across all assets: {len(common_dates)}")
        
        if len(common_dates) < 100:
            print("\nWARNING: Very few common dates found across assets.")
            print("This might indicate data alignment issues or different date ranges.")
        
        # Extract strategy parameters
        entry_rules = strategy_rules['entry_conditions']
        exit_rules = strategy_rules['exit_conditions']
        position_size = strategy_rules['position_size']
        position_type = strategy_rules['position_type']
        
        # Find the primary trading asset (Bitcoin in this case)
        primary_asset_name = None
        for excel_name in asset_data.keys():
            if entry_rules['primary_asset'].lower() in excel_name.lower():
                primary_asset_name = excel_name
                break
        
        if primary_asset_name is None:
            print(f"\nWarning: Could not find primary asset '{entry_rules['primary_asset']}' in data")
            print(f"Available assets: {list(asset_data.keys())}")
            primary_asset_name = list(asset_data.keys())[-1]  # Use last asset as default
            print(f"Using '{primary_asset_name}' as primary asset instead")
        
        primary_data = asset_data[primary_asset_name]
        print(f"\nPrimary trading asset: {primary_asset_name}")
        
        # Track positions
        position = None
        entry_signal_date = None
        
        # Iterate through primary asset data
        for i in range(20, len(primary_data)):
            row = primary_data.iloc[i]
            current_date = row['Date']
            
            # Check for entry signals
            if position is None:
                signal_triggered = False
                
                # For multi-asset strategies, check conditions on all trigger assets
                if ',' in str(entry_rules.get('trigger_asset', '')):
                    # Multiple trigger assets - all must meet condition
                    all_conditions_met = True
                    conditions_log = []
                    
                    for trigger_asset in entry_rules['trigger_asset'].split(','):
                        trigger_asset = trigger_asset.strip()
                        # Find matching asset data
                        trigger_data = None
                        for excel_name, df in asset_data.items():
                            if trigger_asset.lower() in excel_name.lower():
                                trigger_data = df
                                break
                        
                        if trigger_data is not None:
                            # Get data for current date
                            date_data = trigger_data[trigger_data['Date'] == current_date]
                            if not date_data.empty:
                                trigger_row = date_data.iloc[0]
                                # Check condition for this asset
                                condition_met = True
                                
                                # Skip if Return is NaN (first row)
                                if pd.isna(trigger_row['Return']):
                                    all_conditions_met = False
                                    conditions_log.append(f"{trigger_asset}: NO DATA (NaN)")
                                    break
                                
                                if entry_rules['condition'] == 'rises':
                                    # For rises, check if return is positive
                                    condition_met = trigger_row['Return'] > 0
                                elif entry_rules['condition'] == 'drops':
                                    # For drops, check if return is negative
                                    condition_met = trigger_row['Return'] < 0
                                
                                conditions_log.append(f"{trigger_asset}: {'UP' if trigger_row['Return'] > 0 else 'DOWN'} ({trigger_row['Return']:.2%})")
                                
                                if not condition_met:
                                    all_conditions_met = False
                                    break
                            else:
                                all_conditions_met = False
                                conditions_log.append(f"{trigger_asset}: NO DATA")
                                break
                        else:
                            all_conditions_met = False
                            conditions_log.append(f"{trigger_asset}: NOT FOUND")
                            break
                    
                    # Log signal status periodically
                    if all_conditions_met and signal_triggered:
                        print(f"\nSignal on {current_date.strftime('%Y-%m-%d')}: {', '.join(conditions_log)}")
                    
                    signal_triggered = all_conditions_met
                
                else:
                    # Single trigger asset - use existing logic
                    if entry_rules['condition'] == 'drops':
                        if entry_rules['threshold']:
                            signal_triggered = row[f'Price_Change_{entry_rules.get("delay_days", 1)}d'] < -entry_rules['threshold']
                        else:
                            signal_triggered = row['Return'] < 0
                    
                    elif entry_rules['condition'] == 'rises':
                        if entry_rules['threshold']:
                            signal_triggered = row[f'Price_Change_{entry_rules.get("delay_days", 1)}d'] > entry_rules['threshold']
                        else:
                            signal_triggered = row['Return'] > 0
                    
                    elif entry_rules['condition'] == 'crosses_above':
                        if i > 0:
                            prev_row = primary_data.iloc[i-1]
                            signal_triggered = (prev_row['Price'] <= prev_row['SMA_20'] and 
                                              row['Price'] > row['SMA_20'])
                    
                    elif entry_rules['condition'] == 'crosses_below':
                        if i > 0:
                            prev_row = primary_data.iloc[i-1]
                            signal_triggered = (prev_row['Price'] >= prev_row['SMA_20'] and 
                                              row['Price'] < row['SMA_20'])
                
                # Handle entry with delay
                if signal_triggered and entry_signal_date is None:
                    entry_signal_date = row['Date']
                
                if entry_signal_date is not None:
                    days_since_signal = (row['Date'] - entry_signal_date).days
                    
                    if days_since_signal >= entry_rules.get('delay_days', 0):
                        # Enter position
                        position = {
                            'type': position_type,
                            'entry_date': row['Date'],
                            'entry_price': row['Price'],
                            'quantity': (self.capital * position_size) / row['Price'],
                            'signal_date': entry_signal_date,
                            'bars_in_trade': 0,  # Track number of bars
                            'highest_price': row['Price'],  # Track for run-up
                            'lowest_price': row['Price'],   # Track for drawdown
                            'signal_name': f"CEMD_{position_type.title()}"  # Signal name
                        }
                        
                        # Update max contracts tracking
                        self.max_contracts_held = max(self.max_contracts_held, position['quantity'])
                        if position_type == 'long':
                            self.max_contracts_held_long = max(self.max_contracts_held_long, position['quantity'])
                        else:
                            self.max_contracts_held_short = max(self.max_contracts_held_short, position['quantity'])
                        
                        # Enhanced entry logging with position size
                        position_value = position['quantity'] * row['Price']
                        print(f"\nEntering {position_type} position on {row['Date'].strftime('%Y-%m-%d')}:")
                        print(f"  Entry Price: ${row['Price']:.2f}")
                        print(f"  Position Size: {position['quantity']:.6f} units")
                        print(f"  Position Value: ${position_value:,.2f}")
                        print(f"  Capital Used: {position_size * 100:.1f}% (${position_value:,.2f})")
                        
                        entry_signal_date = None
            
            # Check for exit conditions
            else:
                position['bars_in_trade'] += 1  # Increment bar count
                
                # Update highest/lowest prices for run-up/drawdown tracking
                position['highest_price'] = max(position['highest_price'], row['Price'])
                position['lowest_price'] = min(position['lowest_price'], row['Price'])
                
                exit_trade = False
                exit_price = row['Price']
                
                # Time-based exit
                if exit_rules['type'] == 'time_based':
                    days_held = (row['Date'] - position['entry_date']).days
                    if days_held >= exit_rules['value']:
                        exit_trade = True
                
                # Profit target
                elif exit_rules['type'] == 'profit_target':
                    if position['type'] == 'long':
                        profit_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
                        if profit_pct >= exit_rules['value']:
                            exit_trade = True
                    else:  # short
                        profit_pct = ((position['entry_price'] - row['Price']) / position['entry_price']) * 100
                        if profit_pct >= exit_rules['value']:
                            exit_trade = True
                
                # Stop loss
                elif exit_rules['type'] == 'stop_loss':
                    if position['type'] == 'long':
                        loss_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
                        if loss_pct <= -exit_rules['value']:
                            exit_trade = True
                    else:  # short
                        loss_pct = ((position['entry_price'] - row['Price']) / position['entry_price']) * 100
                        if loss_pct <= -exit_rules['value']:
                            exit_trade = True
                
                if exit_trade:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                        # Run-up is highest price - entry price
                        run_up = (position['highest_price'] - position['entry_price']) * position['quantity']
                        run_up_percent = ((position['highest_price'] - position['entry_price']) / position['entry_price']) * 100
                        # Drawdown is entry price - lowest price
                        drawdown = (position['entry_price'] - position['lowest_price']) * position['quantity']
                        drawdown_percent = ((position['entry_price'] - position['lowest_price']) / position['entry_price']) * 100
                    else:  # short
                        pnl = (position['entry_price'] - exit_price) * position['quantity']
                        # For shorts, run-up is entry price - lowest price
                        run_up = (position['entry_price'] - position['lowest_price']) * position['quantity']
                        run_up_percent = ((position['entry_price'] - position['lowest_price']) / position['entry_price']) * 100
                        # Drawdown is highest price - entry price
                        drawdown = (position['highest_price'] - position['entry_price']) * position['quantity']
                        drawdown_percent = ((position['highest_price'] - position['entry_price']) / position['entry_price']) * 100
                    
                    # Update capital
                    self.capital += pnl
                    
                    # Enhanced exit logging
                    pnl_percent = (pnl / (position['entry_price'] * position['quantity'])) * 100
                    print(f"\nExiting position on {row['Date'].strftime('%Y-%m-%d')}:")
                    print(f"  Exit Price: ${exit_price:.2f}")
                    print(f"  Entry Price: ${position['entry_price']:.2f}")
                    print(f"  Position Size: {position['quantity']:.6f} units")
                    print(f"  Price Change: ${exit_price - position['entry_price']:.2f} ({pnl_percent:.2f}%)")
                    print(f"  P&L: ${pnl:,.2f}")
                    print(f"  Days Held: {(row['Date'] - position['entry_date']).days}")
                    
                    # Record trade
                    self.trades.append({
                        'trade_id': len(self.trades) + 1,
                        'type': position['type'],
                        'symbol': primary_asset_name,
                        'signal_date': position['signal_date'],
                        'signal_name': position['signal_name'],
                        'entry_date': position['entry_date'],
                        'exit_date': row['Date'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'cumulative_pnl': self.capital - self.initial_capital,
                        'bars_in_trade': position['bars_in_trade'],
                        'run_up': run_up,
                        'run_up_percent': run_up_percent,
                        'drawdown': drawdown,
                        'drawdown_percent': drawdown_percent
                    })
                    
                    position = None
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Print strategy execution summary
        print(f"\nStrategy Execution Summary:")
        print(f"- Total days analyzed: {len(primary_data) - 20}")  # Subtract initial period
        print(f"- Total trades executed: {len(self.trades)}")
        if len(self.trades) > 0:
            print(f"- First trade: {self.trades[0]['entry_date']}")
            print(f"- Last trade: {self.trades[-1]['exit_date']}")
        else:
            print("- No trades were executed. Check your strategy conditions.")
    
    def _calculate_performance_metrics(self):
        """Calculate all performance metrics"""
        if not self.trades:
            print("No trades executed")
            self.performance_metrics = {
                'total_trades': 0,
                'total_open_trades': 0,
                'net_profit': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'percent_profitable': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_equity_drawdown': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'ratio_avg_win_avg_loss': 0,
                'largest_winning_trade': 0,
                'largest_losing_trade': 0,
                'largest_winning_trade_percent': 0,
                'largest_losing_trade_percent': 0,
                'commission_paid': 0,
                'buy_hold_return': 0,
                'max_equity_runup': 0,
                'avg_pnl': 0,
                'largest_winning_trade_info': None,
                'largest_losing_trade_info': None,
                'avg_trade_duration': 0,
                'avg_bars_in_trades': 0,
                'avg_bars_in_winning_trades': 0,
                'avg_bars_in_losing_trades': 0,
                'sortino_ratio': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_winning_trades': 0,
                'short_winning_trades': 0,
                'long_losing_trades': 0,
                'short_losing_trades': 0,
                'long_percent_profitable': 0,
                'short_percent_profitable': 0,
                'long_avg_pnl': 0,
                'short_avg_pnl': 0,
                'long_avg_winning_trade': 0,
                'short_avg_winning_trade': 0,
                'long_avg_losing_trade': 0,
                'short_avg_losing_trade': 0,
                'max_contracts_held': 0,
                'max_contracts_held_long': 0,
                'max_contracts_held_short': 0,
                'margin_calls': 0,
                'avg_run_up': 0,
                'avg_run_up_percent': 0,
                'avg_drawdown': 0,
                'avg_drawdown_percent': 0,
                'trading_range_start': None,
                'trading_range_end': None,
                'backtesting_range_start': None,
                'backtesting_range_end': None
            }
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Long/Short breakdown
        long_trades = trades_df[trades_df['type'] == 'long']
        short_trades = trades_df[trades_df['type'] == 'short']
        
        long_winning = len(long_trades[long_trades['pnl'] > 0])
        long_losing = len(long_trades[long_trades['pnl'] < 0])
        short_winning = len(short_trades[short_trades['pnl'] > 0])
        short_losing = len(short_trades[short_trades['pnl'] < 0])
        
        # P&L metrics
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        net_profit = gross_profit - gross_loss
        
        # Commission calculation
        commission_rate = self.strategy_properties['commission_rate']
        total_commission = sum([trade['entry_price'] * trade['quantity'] * commission_rate * 2 
                               for trade in self.trades])
        
        # Win/Loss ratios
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Largest trades
        largest_win = trades_df['pnl'].max() if len(trades_df) > 0 else 0
        largest_win_trade = trades_df[trades_df['pnl'] == largest_win].iloc[0] if largest_win > 0 else None
        largest_loss = trades_df['pnl'].min() if len(trades_df) > 0 else 0
        largest_loss_trade = trades_df[trades_df['pnl'] == largest_loss].iloc[0] if largest_loss < 0 else None
        
        # Largest trade percentages
        largest_win_percent = trades_df['pnl_percent'].max() if len(trades_df) > 0 else 0
        largest_loss_percent = trades_df['pnl_percent'].min() if len(trades_df) > 0 else 0
        
        # Duration metrics
        trades_df['duration'] = pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])
        avg_trade_duration = trades_df['duration'].mean()
        
        # Bar metrics
        avg_bars = trades_df['bars_in_trade'].mean() if 'bars_in_trade' in trades_df.columns else 0
        winning_trades_df = trades_df[trades_df['pnl'] > 0]
        losing_trades_df = trades_df[trades_df['pnl'] < 0]
        avg_bars_winning = winning_trades_df['bars_in_trade'].mean() if len(winning_trades_df) > 0 and 'bars_in_trade' in winning_trades_df.columns else 0
        avg_bars_losing = losing_trades_df['bars_in_trade'].mean() if len(losing_trades_df) > 0 and 'bars_in_trade' in losing_trades_df.columns else 0
        
        # Run-up/Drawdown metrics
        avg_run_up = trades_df['run_up'].mean() if 'run_up' in trades_df.columns else 0
        avg_run_up_percent = trades_df['run_up_percent'].mean() if 'run_up_percent' in trades_df.columns else 0
        avg_drawdown = trades_df['drawdown'].mean() if 'drawdown' in trades_df.columns else 0
        avg_drawdown_percent = trades_df['drawdown_percent'].mean() if 'drawdown_percent' in trades_df.columns else 0
        
        # Trading ranges
        trading_range_start = trades_df['entry_date'].min()
        trading_range_end = trades_df['exit_date'].max()
        backtesting_range_start = self.data['Date'].min() if self.data is not None else None
        backtesting_range_end = self.data['Date'].max() if self.data is not None else None
        
        # Long/Short averages
        long_avg_pnl = long_trades['pnl'].mean() if len(long_trades) > 0 else 0
        short_avg_pnl = short_trades['pnl'].mean() if len(short_trades) > 0 else 0
        
        long_winning_trades = long_trades[long_trades['pnl'] > 0]
        long_losing_trades = long_trades[long_trades['pnl'] < 0]
        short_winning_trades = short_trades[short_trades['pnl'] > 0]
        short_losing_trades = short_trades[short_trades['pnl'] < 0]
        
        long_avg_win = long_winning_trades['pnl'].mean() if len(long_winning_trades) > 0 else 0
        long_avg_loss = abs(long_losing_trades['pnl'].mean()) if len(long_losing_trades) > 0 else 0
        short_avg_win = short_winning_trades['pnl'].mean() if len(short_winning_trades) > 0 else 0
        short_avg_loss = abs(short_losing_trades['pnl'].mean()) if len(short_losing_trades) > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_percent'] / 100
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(trades_df['cumulative_pnl'])
        
        # Store all metrics
        self.performance_metrics = {
            'open_pnl': 0,
            'net_profit': net_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'commission_paid': total_commission,
            'buy_hold_return': self._calculate_buy_hold_return(),
            'max_equity_runup': trades_df['cumulative_pnl'].max() if len(trades_df) > 0 else 0,
            'max_equity_drawdown': max_drawdown,
            'total_trades': total_trades,
            'total_open_trades': 0,  # All trades are closed in backtest
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'percent_profitable': win_rate,
            'avg_pnl': net_profit / total_trades if total_trades > 0 else 0,
            'avg_winning_trade': avg_win,
            'avg_losing_trade': avg_loss,
            'ratio_avg_win_avg_loss': avg_win_loss_ratio,
            'largest_winning_trade': largest_win,
            'largest_winning_trade_percent': largest_win_percent,
            'largest_winning_trade_info': largest_win_trade,
            'largest_losing_trade': largest_loss,
            'largest_losing_trade_percent': largest_loss_percent,
            'largest_losing_trade_info': largest_loss_trade,
            'avg_trade_duration': avg_trade_duration.days if pd.notna(avg_trade_duration) else 0,
            'avg_bars_in_trades': avg_bars,
            'avg_bars_in_winning_trades': avg_bars_winning,
            'avg_bars_in_losing_trades': avg_bars_losing,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            # Long/Short specific metrics
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_winning_trades': long_winning,
            'short_winning_trades': short_winning,
            'long_losing_trades': long_losing,
            'short_losing_trades': short_losing,
            'long_percent_profitable': (long_winning / len(long_trades) * 100) if len(long_trades) > 0 else 0,
            'short_percent_profitable': (short_winning / len(short_trades) * 100) if len(short_trades) > 0 else 0,
            'long_avg_pnl': long_avg_pnl,
            'short_avg_pnl': short_avg_pnl,
            'long_avg_winning_trade': long_avg_win,
            'short_avg_winning_trade': short_avg_win,
            'long_avg_losing_trade': long_avg_loss,
            'short_avg_losing_trade': short_avg_loss,
            # New metrics from CSVs
            'max_contracts_held': self.max_contracts_held,
            'max_contracts_held_long': self.max_contracts_held_long,
            'max_contracts_held_short': self.max_contracts_held_short,
            'margin_calls': 0,  # Always 0 in backtest
            'avg_run_up': avg_run_up,
            'avg_run_up_percent': avg_run_up_percent,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_percent': avg_drawdown_percent,
            'trading_range_start': trading_range_start,
            'trading_range_end': trading_range_end,
            'backtesting_range_start': backtesting_range_start,
            'backtesting_range_end': backtesting_range_end
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_pnl) == 0:
            return 0
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + self.initial_capital)
        return drawdown.min() * 100
    
    def _calculate_buy_hold_return(self) -> float:
        """Calculate buy and hold return"""
        if self.data is None or len(self.data) == 0:
            return 0
        
        # Get primary asset data (Bitcoin)
        bitcoin_data = self.data[self.data['excel_name'].str.contains('Bitcoin', case=False)]
        if len(bitcoin_data) == 0:
            return 0
            
        first_price = bitcoin_data.iloc[0]['Close']
        last_price = bitcoin_data.iloc[-1]['Close']
        return ((last_price - first_price) / first_price) * self.initial_capital
    
    def generate_report(self, save_path: str = 'backtest_report.html'):
        """Generate comprehensive HTML report with all metrics and visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cryptocurrency Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #1a1a1a;
                    color: #e0e0e0;
                    margin: 20px;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #2a2a2a;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .metric-title {{
                    font-size: 14px;
                    color: #888;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #fff;
                }}
                .positive {{
                    color: #4caf50;
                }}
                .negative {{
                    color: #f44336;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #444;
                }}
                th {{
                    background-color: #333;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Cryptocurrency Backtest Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Trading Range: {self.performance_metrics['trading_range_start'].strftime('%Y-%m-%d') if self.performance_metrics['trading_range_start'] else 'N/A'} to {self.performance_metrics['trading_range_end'].strftime('%Y-%m-%d') if self.performance_metrics['trading_range_end'] else 'N/A'}</p>
                    <p>Backtesting Range: {self.performance_metrics['backtesting_range_start'].strftime('%Y-%m-%d') if self.performance_metrics['backtesting_range_start'] else 'N/A'} to {self.performance_metrics['backtesting_range_end'].strftime('%Y-%m-%d') if self.performance_metrics['backtesting_range_end'] else 'N/A'}</p>
                </div>
                
                <h2>Performance Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Net Profit</div>
                        <div class="metric-value {'positive' if self.performance_metrics['net_profit'] >= 0 else 'negative'}">
                            ${self.performance_metrics['net_profit']:,.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Trades</div>
                        <div class="metric-value">{self.performance_metrics['total_trades']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Win Rate</div>
                        <div class="metric-value">{self.performance_metrics['percent_profitable']:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Profit Factor</div>
                        <div class="metric-value">{self.performance_metrics['profit_factor']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Max Contracts Held</div>
                        <div class="metric-value">{self.performance_metrics['max_contracts_held']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Margin Calls</div>
                        <div class="metric-value">{self.performance_metrics['margin_calls']}</div>
                    </div>
                </div>
                
                <h2>Trade Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Gross Profit</div>
                        <div class="metric-value positive">${self.performance_metrics['gross_profit']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Gross Loss</div>
                        <div class="metric-value negative">${self.performance_metrics['gross_loss']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Commission Paid</div>
                        <div class="metric-value">${self.performance_metrics['commission_paid']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Max Drawdown</div>
                        <div class="metric-value negative">{self.performance_metrics['max_equity_drawdown']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg # Bars in Trades</div>
                        <div class="metric-value">{self.performance_metrics['avg_bars_in_trades']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg # Bars in Winning Trades</div>
                        <div class="metric-value">{self.performance_metrics['avg_bars_in_winning_trades']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg # Bars in Losing Trades</div>
                        <div class="metric-value">{self.performance_metrics['avg_bars_in_losing_trades']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Largest Winning Trade %</div>
                        <div class="metric-value positive">{self.performance_metrics['largest_winning_trade_percent']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Largest Losing Trade %</div>
                        <div class="metric-value negative">{self.performance_metrics['largest_losing_trade_percent']:.2f}%</div>
                    </div>
                </div>
                
                <h2>Run-up/Drawdown Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Avg Run-up</div>
                        <div class="metric-value positive">${self.performance_metrics['avg_run_up']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Run-up %</div>
                        <div class="metric-value positive">{self.performance_metrics['avg_run_up_percent']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Drawdown</div>
                        <div class="metric-value negative">${self.performance_metrics['avg_drawdown']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Drawdown %</div>
                        <div class="metric-value negative">{self.performance_metrics['avg_drawdown_percent']:.2f}%</div>
                    </div>
                </div>
                
                <h2>Long/Short Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Long Trades</div>
                        <div class="metric-value">{self.performance_metrics['long_trades']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Short Trades</div>
                        <div class="metric-value">{self.performance_metrics['short_trades']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Long Win Rate</div>
                        <div class="metric-value">{self.performance_metrics['long_percent_profitable']:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Short Win Rate</div>
                        <div class="metric-value">{self.performance_metrics['short_percent_profitable']:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Long Avg P&L</div>
                        <div class="metric-value {'positive' if self.performance_metrics['long_avg_pnl'] >= 0 else 'negative'}">${self.performance_metrics['long_avg_pnl']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Short Avg P&L</div>
                        <div class="metric-value {'positive' if self.performance_metrics['short_avg_pnl'] >= 0 else 'negative'}">${self.performance_metrics['short_avg_pnl']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Max Long Contracts</div>
                        <div class="metric-value">{self.performance_metrics['max_contracts_held_long']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Max Short Contracts</div>
                        <div class="metric-value">{self.performance_metrics['max_contracts_held_short']:.2f}</div>
                    </div>
                </div>
                
                <h2>Risk/Performance Ratios</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Sharpe Ratio</div>
                        <div class="metric-value">{self.performance_metrics['sharpe_ratio']:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sortino Ratio</div>
                        <div class="metric-value">{self.performance_metrics['sortino_ratio']:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Win/Loss Ratio</div>
                        <div class="metric-value">{self.performance_metrics['ratio_avg_win_avg_loss']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Buy & Hold Return</div>
                        <div class="metric-value">${self.performance_metrics['buy_hold_return']:,.2f}</div>
                    </div>
                </div>
                
                <h2>Trade List</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Trade #</th>
                            <th>Type</th>
                            <th>Signal</th>
                            <th>Symbol</th>
                            <th>Entry Date</th>
                            <th>Exit Date</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>Quantity</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Run-up %</th>
                            <th>Drawdown %</th>
                            <th>Bars</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add trade rows
        for trade in self.trades[-20:]:  # Show last 20 trades
            pnl_class = 'positive' if trade['pnl'] >= 0 else 'negative'
            html_content += f"""
                        <tr>
                            <td>{trade['trade_id']}</td>
                            <td>{trade['type'].upper()}</td>
                            <td>{trade.get('signal_name', 'N/A')}</td>
                            <td>{trade['symbol']}</td>
                            <td>{trade['entry_date'].strftime('%Y-%m-%d') if isinstance(trade['entry_date'], pd.Timestamp) else trade['entry_date']}</td>
                            <td>{trade['exit_date'].strftime('%Y-%m-%d') if isinstance(trade['exit_date'], pd.Timestamp) else trade['exit_date']}</td>
                            <td>${trade['entry_price']:.2f}</td>
                            <td>${trade['exit_price']:.2f}</td>
                            <td>{trade['quantity']:.4f}</td>
                            <td class="{pnl_class}">${trade['pnl']:,.2f}</td>
                            <td class="{pnl_class}">{trade['pnl_percent']:.2f}%</td>
                            <td class="positive">{trade.get('run_up_percent', 0):.2f}%</td>
                            <td class="negative">{trade.get('drawdown_percent', 0):.2f}%</td>
                            <td>{trade.get('bars_in_trade', 'N/A')}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {save_path}")
    
    def plot_equity_curve(self):
        """Plot interactive equity curve using Plotly with error handling"""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Create figure
            fig = go.Figure()
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=trades_df['exit_date'],
                y=trades_df['cumulative_pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(color=self.colors['green'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 215, 117, 0.1)'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            # Mark best and worst trades
            if len(trades_df) > 0:
                best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
                worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
                
                fig.add_annotation(
                    x=best_trade['exit_date'],
                    y=best_trade['cumulative_pnl'],
                    text=f"Best Trade<br>${best_trade['pnl']:,.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=self.colors['green'],
                    ax=0,
                    ay=-40,
                    bgcolor=self.colors['green'],
                    opacity=0.8
                )
                
                if worst_trade['pnl'] < 0:  # Only show worst trade if it's negative
                    fig.add_annotation(
                        x=worst_trade['exit_date'],
                        y=worst_trade['cumulative_pnl'],
                        text=f"Worst Trade<br>${worst_trade['pnl']:,.0f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=self.colors['red'],
                        ax=0,
                        ay=40,
                        bgcolor=self.colors['red'],
                        opacity=0.8
                    )
            
            # Add drawdown shading
            running_max = trades_df['cumulative_pnl'].expanding().max()
            drawdown = trades_df['cumulative_pnl'] - running_max
            
            fig.add_trace(go.Scatter(
                x=trades_df['exit_date'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.colors['red'], width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 51, 102, 0.2)',
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Equity Curve - Total Return: ${self.performance_metrics["net_profit"]:,.2f} ({(self.performance_metrics["net_profit"]/self.initial_capital*100):.1f}%)',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Date',
                yaxis_title='Cumulative P&L ($)',
                yaxis2=dict(
                    title='Drawdown ($)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                template='plotly_dark',
                showlegend=True,
                height=600,
                xaxis_rangeslider_visible=True
            )
            
            # Add performance metrics box
            metrics_text = f"""
            Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}
            Max Drawdown: {self.performance_metrics['max_equity_drawdown']:.1f}%
            Win Rate: {self.performance_metrics['percent_profitable']:.1f}%
            Profit Factor: {self.performance_metrics['profit_factor']:.2f}
            """
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=metrics_text,
                showarrow=False,
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                borderwidth=1,
                font=dict(size=10, color="white"),
                align="left",
                xanchor="left",
                yanchor="top"
            )
            
            # Save to HTML file only - no display
            output_file = 'equity_curve_interactive.html'
            fig.write_html(output_file)
            print(f"Interactive equity curve saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating equity curve: {e}")
            print("Attempting simple matplotlib fallback...")
            
            # Fallback to simple matplotlib plot
            try:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                trades_df = pd.DataFrame(self.trades)
                ax.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], 
                       color=self.colors['green'], linewidth=2, label='Equity Curve')
                ax.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                               color=self.colors['green'], alpha=0.3)
                ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                
                ax.set_title(f'Equity Curve - Total Return: ${self.performance_metrics["net_profit"]:,.2f}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative P&L ($)')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('equity_curve_simple.png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
                plt.close()
                print("Simple equity curve saved to equity_curve_simple.png")
                
            except Exception as e2:
                print(f"Fallback plot also failed: {e2}")
    
    def plot_trade_distribution(self):
        """Plot professional trade distribution analysis with error handling"""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Create subplots with plotly
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'P&L Distribution', 'Win Rate by Entry Time', 'Risk vs Reward',
                    'Trade Duration vs Profit', 'Cumulative Win Rate', 'P&L by Trade Number'
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            
            # 1. P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=trades_df['pnl'],
                    nbinsx=30,
                    name='P&L Distribution',
                    marker_color=trades_df['pnl'].apply(lambda x: self.colors['green'] if x > 0 else self.colors['red']),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Win Rate by Entry Time (simplified)
            trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.month
            win_by_month = trades_df.groupby('entry_month').agg({
                'pnl': lambda x: (x > 0).mean() * 100
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=win_by_month['entry_month'],
                    y=win_by_month['pnl'],
                    showlegend=False,
                    marker_color=self.colors['blue']
                ),
                row=1, col=2
            )
            
            # 3. Risk vs Reward Scatter
            fig.add_trace(
                go.Scatter(
                    x=trades_df.get('drawdown_percent', [0] * len(trades_df)),
                    y=trades_df.get('run_up_percent', [0] * len(trades_df)),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=trades_df['pnl'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L $", x=0.68)
                    ),
                    showlegend=False
                ),
                row=1, col=3
            )
            
            # 4. Trade Duration vs Profit
            trades_df['duration_days'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df['duration_days'],
                    y=trades_df['pnl_percent'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=trades_df['pnl_percent'],
                        colorscale='RdYlGn',
                        showscale=False
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 5. Cumulative Win Rate
            trades_df['cumulative_win_rate'] = (trades_df['pnl'] > 0).expanding().mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df['trade_id'],
                    y=trades_df['cumulative_win_rate'],
                    mode='lines',
                    line=dict(color=self.colors['blue'], width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Add 50% reference line
            fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5, row=2, col=2)
            
            # 6. P&L by Trade Number
            colors = [self.colors['green'] if x > 0 else self.colors['red'] for x in trades_df['pnl']]
            
            fig.add_trace(
                go.Bar(
                    x=trades_df['trade_id'],
                    y=trades_df['pnl'],
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                title_text='Professional Trade Analysis Dashboard',
                showlegend=False,
                template='plotly_dark',
                height=800
            )
            
            # Update axes
            fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
            fig.update_xaxes(title_text="Month", row=1, col=2)
            fig.update_xaxes(title_text="Max Drawdown %", row=1, col=3)
            fig.update_xaxes(title_text="Duration (Days)", row=2, col=1)
            fig.update_xaxes(title_text="Trade Number", row=2, col=2)
            fig.update_xaxes(title_text="Trade Number", row=2, col=3)
            
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
            fig.update_yaxes(title_text="Max Run-up %", row=1, col=3)
            fig.update_yaxes(title_text="Return %", row=2, col=1)
            fig.update_yaxes(title_text="Win Rate %", row=2, col=2)
            fig.update_yaxes(title_text="P&L ($)", row=2, col=3)
            
            output_file = 'trade_distribution_interactive.html'
            fig.write_html(output_file)
            print(f"Interactive trade distribution saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating trade distribution: {e}")
            
            # Fallback to simple matplotlib plots
            try:
                plt.style.use('dark_background')
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                trades_df = pd.DataFrame(self.trades)
                
                # 1. P&L Distribution
                ax = axes[0, 0]
                wins = trades_df[trades_df['pnl'] > 0]['pnl']
                losses = trades_df[trades_df['pnl'] < 0]['pnl']
                
                if len(wins) > 0:
                    ax.hist(wins, bins=20, alpha=0.7, label='Wins', color=self.colors['green'])
                if len(losses) > 0:
                    ax.hist(losses, bins=20, alpha=0.7, label='Losses', color=self.colors['red'])
                ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
                ax.set_title('P&L Distribution')
                ax.set_xlabel('P&L ($)')
                ax.legend()
                
                # 2. Returns Box Plot
                ax = axes[0, 1]
                if 'pnl_percent' in trades_df.columns:
                    trades_df.boxplot(column='pnl_percent', by='type', ax=ax)
                    ax.set_title('Returns by Position Type')
                    ax.set_ylabel('Return %')
                
                # 3. Cumulative P&L
                ax = axes[1, 0]
                ax.plot(trades_df['trade_id'], trades_df['cumulative_pnl'], 
                       color=self.colors['green'], linewidth=2)
                ax.fill_between(trades_df['trade_id'], 0, trades_df['cumulative_pnl'], 
                               where=trades_df['cumulative_pnl'] >= 0, 
                               color=self.colors['green'], alpha=0.3)
                ax.fill_between(trades_df['trade_id'], 0, trades_df['cumulative_pnl'], 
                               where=trades_df['cumulative_pnl'] < 0, 
                               color=self.colors['red'], alpha=0.3)
                ax.set_title('Cumulative P&L')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Cumulative P&L ($)')
                
                # 4. Win Rate
                ax = axes[1, 1]
                trades_df['cumulative_win_rate'] = (trades_df['pnl'] > 0).expanding().mean() * 100
                ax.plot(trades_df['trade_id'], trades_df['cumulative_win_rate'], 
                       color=self.colors['blue'], linewidth=2)
                ax.axhline(y=50, color='white', linestyle='--', alpha=0.5)
                ax.set_title('Cumulative Win Rate')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Win Rate (%)')
                
                plt.tight_layout()
                plt.savefig('trade_analysis_simple.png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
                plt.close()
                print("Simple trade analysis saved to trade_analysis_simple.png")
                
            except Exception as e2:
                print(f"Fallback plot also failed: {e2}")
    
    def plot_price_chart_with_trades(self, num_days: int = 100):
        """Plot professional candlestick chart with trades using mplfinance"""
        if self.data is None:
            print("No data to plot")
            return
            
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            # Get Bitcoin data for plotting
            bitcoin_data = self.data[self.data['excel_name'].str.contains('Bitcoin', case=False)].copy()
            if len(bitcoin_data) == 0:
                print("No Bitcoin data found for plotting")
                return
            
            # Prepare OHLC data (using Close for all since we only have Close)
            bitcoin_data['Open'] = bitcoin_data['Close']
            bitcoin_data['High'] = bitcoin_data['Close'] * 1.001  # Slight variation for visibility
            bitcoin_data['Low'] = bitcoin_data['Close'] * 0.999
            bitcoin_data['Volume'] = 1000000  # Dummy volume
            
            # Set Date as index
            bitcoin_data.set_index('Date', inplace=True)
            
            # Get last num_days of data
            plot_data = bitcoin_data.tail(num_days).copy()
            
            # Calculate simple indicators
            plot_data['SMA_20'] = plot_data['Close'].rolling(window=20).mean()
            plot_data['SMA_50'] = plot_data['Close'].rolling(window=50).mean()
            
            # Define custom style
            mc = mpf.make_marketcolors(
                up='#00D775',
                down='#FF3366',
                edge='inherit',
                wick={'up': '#00D775', 'down': '#FF3366'},
                volume='inherit'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                style_name='professional',
                y_on_right=True,
                gridstyle='-',
                gridcolor='#262730',
                facecolor='#0e1117',
                edgecolor='#0e1117',
                figcolor='#0e1117',
                rc={'font.size': 10}
            )
            
            # Create additional plots for moving averages
            additional_plots = []
            if not plot_data['SMA_20'].isna().all():
                additional_plots.append(
                    mpf.make_addplot(plot_data['SMA_20'], color='yellow', width=1.5)
                )
            if not plot_data['SMA_50'].isna().all():
                additional_plots.append(
                    mpf.make_addplot(plot_data['SMA_50'], color='orange', width=1.5)
                )
            
            # Plot with mplfinance
            kwargs = {
                'type': 'candle',
                'style': s,
                'volume': True,
                'figsize': (14, 8),
                'title': f'\nBitcoin Price Chart - Last {num_days} Days\n',
                'savefig': 'bitcoin_price_chart.png'
            }
            
            if additional_plots:
                kwargs['addplot'] = additional_plots
            
            mpf.plot(plot_data, **kwargs)
            print("Price chart saved to bitcoin_price_chart.png")
            
        except Exception as e:
            print(f"Error creating price chart: {e}")
            
            # Fallback to simple matplotlib
            try:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                bitcoin_data = self.data[self.data['excel_name'].str.contains('Bitcoin', case=False)]
                if len(bitcoin_data) > 0:
                    plot_data = bitcoin_data.tail(num_days)
                    ax.plot(plot_data['Date'], plot_data['Close'], 
                           color=self.colors['green'], linewidth=2, label='Bitcoin Price')
                    
                    # Mark trades
                    trades_df = pd.DataFrame(self.trades)
                    for _, trade in trades_df.iterrows():
                        if trade['entry_date'] in plot_data['Date'].values:
                            entry_price = plot_data[plot_data['Date'] == trade['entry_date']]['Close'].values[0]
                            if trade['type'] == 'long':
                                ax.scatter(trade['entry_date'], entry_price, 
                                         color=self.colors['green'], marker='^', s=100, zorder=5)
                            else:
                                ax.scatter(trade['entry_date'], entry_price, 
                                         color=self.colors['red'], marker='v', s=100, zorder=5)
                    
                    ax.set_title(f'Bitcoin Price - Last {num_days} Days')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price ($)')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    plt.tight_layout()
                    plt.savefig('bitcoin_price_simple.png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
                    plt.close()
                    print("Simple price chart saved to bitcoin_price_simple.png")
                
            except Exception as e2:
                print(f"Fallback plot also failed: {e2}")
    
    def plot_comprehensive_dashboard(self):
        """Create a simplified comprehensive dashboard"""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Create a simple summary plot with matplotlib
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(16, 10))
            
            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Equity Curve
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], 
                    color=self.colors['green'], linewidth=2)
            ax1.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                           color=self.colors['green'], alpha=0.3)
            ax1.set_title('Equity Curve', fontsize=14)
            ax1.set_ylabel('Cumulative P&L ($)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Key Metrics
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            metrics_text = f"""Key Metrics:
            
Net Profit: ${self.performance_metrics['net_profit']:,.2f}
Win Rate: {self.performance_metrics['percent_profitable']:.1f}%
Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}
Max Drawdown: {self.performance_metrics['max_equity_drawdown']:.1f}%
Total Trades: {self.performance_metrics['total_trades']}
Profit Factor: {self.performance_metrics['profit_factor']:.2f}"""
            
            ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            # 3. P&L Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] < 0]['pnl']
            
            if len(wins) > 0:
                ax3.hist(wins, bins=20, alpha=0.7, color=self.colors['green'], label='Wins')
            if len(losses) > 0:
                ax3.hist(losses, bins=10, alpha=0.7, color=self.colors['red'], label='Losses')
            ax3.set_title('P&L Distribution')
            ax3.set_xlabel('P&L ($)')
            ax3.legend()
            
            # 4. Monthly Returns
            ax4 = fig.add_subplot(gs[1, 1])
            trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_returns = trades_df.groupby('month')['pnl'].sum()
            
            if len(monthly_returns) > 0:
                colors = [self.colors['green'] if x > 0 else self.colors['red'] for x in monthly_returns]
                ax4.bar(range(len(monthly_returns)), monthly_returns.values, color=colors)
                ax4.set_title('Monthly P&L')
                ax4.set_xlabel('Month')
                ax4.set_ylabel('P&L ($)')
            
            # 5. Win Rate Evolution
            ax5 = fig.add_subplot(gs[1, 2])
            trades_df['cumulative_win_rate'] = (trades_df['pnl'] > 0).expanding().mean() * 100
            ax5.plot(trades_df['trade_id'], trades_df['cumulative_win_rate'], 
                    color=self.colors['blue'], linewidth=2)
            ax5.axhline(y=50, color='white', linestyle='--', alpha=0.5)
            ax5.set_title('Cumulative Win Rate')
            ax5.set_ylabel('Win Rate (%)')
            
            # 6. Trade P&L
            ax6 = fig.add_subplot(gs[2, :])
            colors = [self.colors['green'] if x > 0 else self.colors['red'] for x in trades_df['pnl']]
            ax6.bar(trades_df['trade_id'], trades_df['pnl'], color=colors, alpha=0.7)
            ax6.set_title('Individual Trade P&L')
            ax6.set_xlabel('Trade Number')
            ax6.set_ylabel('P&L ($)')
            ax6.grid(True, alpha=0.3)
            
            plt.suptitle('Trading Performance Dashboard', fontsize=16)
            plt.tight_layout()
            plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
            plt.close()
            print("Comprehensive dashboard saved to comprehensive_dashboard.png")
            
            # Also try to create HTML version with plotly
            self._create_plotly_dashboard_safe()
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
    
    def _create_plotly_dashboard_safe(self):
        """Create a safe version of plotly dashboard that won't throw errors"""
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Create a simple plotly figure
            fig = go.Figure()
            
            # Just add equity curve
            fig.add_trace(go.Scatter(
                x=trades_df['exit_date'],
                y=trades_df['cumulative_pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(color=self.colors['green'], width=2)
            ))
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L ($)',
                template='plotly_dark',
                height=600
            )
            
            # Save without showing
            output_file = 'equity_curve_plotly.html'
            fig.write_html(output_file)
            print(f"Plotly equity curve saved to {output_file}")
            
        except Exception as e:
            print(f"Plotly dashboard creation failed: {e}")
    
    def plot_price_data(self, num_days: int = 100):
        """Plot price data for all assets"""
        if self.data is None:
            print("No data to plot")
            return
        
        try:
            # Create subplots for each asset
            unique_assets = self.data['excel_name'].unique()
            fig, axes = plt.subplots(len(unique_assets), 1, figsize=(12, 4 * len(unique_assets)))
            plt.style.use('dark_background')
            
            if len(unique_assets) == 1:
                axes = [axes]
            
            for i, asset_name in enumerate(unique_assets):
                asset_data = self.data[self.data['excel_name'] == asset_name].tail(num_days)
                
                axes[i].plot(asset_data['Date'], asset_data['Price'], 'w-', linewidth=1)
                axes[i].set_title(f'{asset_name}', fontsize=14)
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
                
                if i == len(unique_assets) - 1:
                    axes[i].set_xlabel('Date')
                    
                # Rotate x-axis labels
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('price_data.png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
            plt.close()
            print("Price data saved to price_data.png")
            
        except Exception as e:
            print(f"Error plotting price data: {e}")
    
    def print_summary(self):
        """Print performance summary to console"""
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nOverview:")
        print(f"  Net Profit: ${self.performance_metrics['net_profit']:,.2f}")
        print(f"  Total Trades: {self.performance_metrics['total_trades']}")
        print(f"  Win Rate: {self.performance_metrics['percent_profitable']:.1f}%")
        print(f"  Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        print(f"  Max Contracts Held: {self.performance_metrics['max_contracts_held']:.2f}")
        
        print(f"\nTrade Analysis:")
        print(f"  Winning Trades: {self.performance_metrics['winning_trades']}")
        print(f"  Losing Trades: {self.performance_metrics['losing_trades']}")
        print(f"  Avg Win: ${self.performance_metrics['avg_winning_trade']:,.2f}")
        print(f"  Avg Loss: ${self.performance_metrics['avg_losing_trade']:,.2f}")
        print(f"  Largest Win: ${self.performance_metrics['largest_winning_trade']:,.2f}")
        print(f"  Largest Loss: ${self.performance_metrics['largest_losing_trade']:,.2f}")
        print(f"  Avg # Bars in Trades: {self.performance_metrics['avg_bars_in_trades']:.1f}")
        
        print(f"\nRun-up/Drawdown:")
        print(f"  Avg Run-up: ${self.performance_metrics['avg_run_up']:,.2f} ({self.performance_metrics['avg_run_up_percent']:.2f}%)")
        print(f"  Avg Drawdown: ${self.performance_metrics['avg_drawdown']:,.2f} ({self.performance_metrics['avg_drawdown_percent']:.2f}%)")
        
        print(f"\nLong/Short Breakdown:")
        print(f"  Long Trades: {self.performance_metrics['long_trades']} ({self.performance_metrics['long_percent_profitable']:.1f}% profitable)")
        print(f"  Short Trades: {self.performance_metrics['short_trades']} ({self.performance_metrics['short_percent_profitable']:.1f}% profitable)")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {self.performance_metrics['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: {self.performance_metrics['max_equity_drawdown']:.2f}%")
        print(f"  Commission Paid: ${self.performance_metrics['commission_paid']:,.2f}")
        print(f"  Margin Calls: {self.performance_metrics['margin_calls']}")
        
        print("\n" + "="*60)

# Main execution function
def run_backtest(strategy_description: str, excel_names: List[str], 
                pinecone_api_key: str = None, openai_api_key: str = None,
                initial_capital: float = 100000.0):
    """
    Main function to run the complete backtest with natural language strategy
    
    Parameters:
    - strategy_description: Natural language description of the strategy
    - excel_names: List of excel names to fetch from Pinecone
    - pinecone_api_key: Pinecone API key (optional, will use env var if not provided)
    - openai_api_key: OpenAI API key (optional, will use env var if not provided)
    - initial_capital: Starting capital for backtest
    """
    
    print(f"Strategy: {strategy_description}")
    print(f"Excel names: {excel_names}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    
    # Initialize backtester (will use env vars if keys not provided)
    backtester = CryptoBacktester(pinecone_api_key, openai_api_key)
    
    # Parse strategy using AI
    print("\nParsing strategy with AI...")
    strategy_rules = backtester.parse_strategy_with_ai(strategy_description)
    
    # Fetch data from Pinecone
    print("\nFetching data from Pinecone...")
    data = backtester.fetch_data_from_pinecone(excel_names)
    print(f"Loaded {len(data)} records")
    
    # Run custom strategy
    print("\nRunning custom strategy...")
    backtester.run_custom_strategy(
        strategy_rules=strategy_rules,
        initial_capital=initial_capital
    )
    
    # Generate visualizations
    print("\nGenerating professional visualizations...")
    if len(backtester.trades) > 0:
        try:
            backtester.plot_equity_curve()
            backtester.plot_trade_distribution()
            backtester.plot_price_chart_with_trades()
            backtester.plot_comprehensive_dashboard()
        except Exception as e:
            print(f"Note: Some interactive plots couldn't be displayed in notebook: {e}")
            print("But all charts have been saved as HTML/PNG files!")
    else:
        print("No trades executed - plotting price data instead")
        # Still plot price data to see what happened
        backtester.plot_price_data()
    
    # Generate report
    print("\nGenerating HTML report...")
    backtester.generate_report()
    
    # Print summary
    backtester.print_summary()
    
    return backtester
