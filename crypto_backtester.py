"""
Cryptocurrency Backtesting Framework
A sophisticated AI-powered backtesting system for cryptocurrency trading strategies.
"""

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

# Auto-install required libraries
def install_and_import(package_name: str, import_name: str = None):
    """Helper function to install and import packages"""
    if import_name is None:
        import_name = package_name
    
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call(['pip', 'install', package_name, '--quiet'])
        return __import__(import_name)

# Import visualization libraries with auto-installation
mpf = install_and_import('mplfinance', 'mplfinance')
go = install_and_import('plotly', 'plotly.graph_objects').graph_objects
make_subplots = install_and_import('plotly', 'plotly.subplots').subplots.make_subplots
px = install_and_import('plotly', 'plotly.express').express
ta = install_and_import('ta', 'ta')


class CryptoBacktester:
    """
    A comprehensive cryptocurrency backtesting framework with AI-powered strategy parsing.
    
    This class provides functionality to:
    - Parse natural language trading strategies using OpenAI
    - Fetch historical data from Pinecone vector database
    - Execute backtests with detailed position tracking
    - Generate professional visualizations and reports
    - Track comprehensive performance metrics
    
    Attributes:
        pinecone_api_key (str): API key for Pinecone
        openai_api_key (str): API key for OpenAI
        index_name (str): Name of the Pinecone index
        data (pd.DataFrame): Historical price data
        trades (List[Dict]): List of executed trades
        performance_metrics (Dict): Calculated performance metrics
    """
    
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None):
        """
        Initialize the backtesting framework with API connections.
        
        Args:
            pinecone_api_key (str, optional): Pinecone API key. If not provided, uses environment variable.
            openai_api_key (str, optional): OpenAI API key. If not provided, uses environment variable.
            
        Raises:
            ValueError: If API keys are not found in parameters or environment variables.
        """
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
        
        # Initialize OpenAI
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
            'green': '#00D775',
            'red': '#FF3366',
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'background': '#0e1117',
            'grid': '#262730'
        }
        
    def parse_strategy_with_ai(self, strategy_description: str) -> Dict:
        """
        Parse natural language trading strategy into structured rules using AI.
        
        Args:
            strategy_description (str): Natural language description of the trading strategy
            
        Returns:
            Dict: Structured strategy rules including entry/exit conditions and position sizing
            
        Example:
            >>> rules = backtester.parse_strategy_with_ai("Buy Bitcoin when it drops 5% and hold for 10 days")
            >>> print(rules)
            {
                "entry_conditions": {...},
                "exit_conditions": {...},
                "position_size": 0.1,
                "position_type": "long"
            }
        """
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
        """Parse the raw_text field from Pinecone to extract key-value pairs."""
        data = {}
        parts = raw_text.split(' | ')
        
        for part in parts:
            if ': ' in part:
                try:
                    key, value = part.split(': ', 1)
                    if 'Date' in key:
                        data['Date'] = value
                    elif 'Symbol' in key:
                        data['Symbol'] = value
                    else:
                        clean_value = value.replace('$', '').replace(',', '').replace('%', '').strip()
                        try:
                            numeric_value = float(clean_value)
                            if 'Close' not in data and numeric_value != 0:
                                data['Close'] = numeric_value
                                if 'Yield' in key:
                                    data['is_yield'] = True
                        except ValueError:
                            continue
                except ValueError:
                    continue
        
        # Infer symbol if not found
        if 'Symbol' not in data:
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
        """
        Fetch historical data from Pinecone vector database.
        
        Args:
            excel_names (List[str]): List of data series names to fetch from Pinecone
            
        Returns:
            pd.DataFrame: Combined dataframe with all requested data series
            
        Raises:
            ValueError: If no data is found for the specified excel names
        """
        all_data = []
        
        for i, excel_name in enumerate(excel_names):
            print(f"\nFetching data for: {excel_name}")
            query_response = self.index.query(
                vector=[0.0] * 1536,
                filter={"excel_name": {"$eq": excel_name}},
                top_k=10000,
                include_metadata=True
            )
            
            print(f"Found {len(query_response['matches'])} records")
            
            for match in query_response['matches']:
                metadata = match['metadata']
                raw_text = metadata.get('raw_text', '')
                data_dict = self._parse_raw_text(raw_text)
                
                if 'Close' not in data_dict or 'Date' not in data_dict:
                    continue
                    
                data_dict.update({
                    'id': match['id'],
                    'excel_name': metadata.get('excel_name'),
                    'filename': metadata.get('filename'),
                    'sheet': metadata.get('sheet'),
                    'row_idx': metadata.get('row_idx')
                })
                all_data.append(data_dict)
        
        df = pd.DataFrame(all_data)
        if len(df) == 0:
            raise ValueError(f"No data found for excel names: {excel_names}")
            
        # Convert and clean data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        print(f"\nTotal records loaded: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        self.data = df
        return df
    
    def add_basic_indicators(self):
        """Add technical indicators to the data."""
        if self.data is None:
            raise ValueError("No data loaded. Please fetch data first.")
        
        self.data['Price'] = self.data['Close']
        self.data['Return'] = self.data['Price'].pct_change()
        self.data['Cumulative_Return'] = (1 + self.data['Return']).cumprod()
        
        # Moving averages
        self.data['SMA_5'] = self.data['Price'].rolling(window=5).mean()
        self.data['SMA_20'] = self.data['Price'].rolling(window=20).mean()
        
        # Price changes
        for days in [1, 3, 5, 7, 10]:
            self.data[f'Price_Change_{days}d'] = self.data['Price'].pct_change(days) * 100
            
        print(f"\nIndicators added. Data shape: {self.data.shape}")
    
    def run_custom_strategy(self, strategy_rules: Dict, initial_capital: float = 100000.0):
        """
        Execute a backtest based on parsed strategy rules.
        
        Args:
            strategy_rules (Dict): Parsed strategy rules from parse_strategy_with_ai()
            initial_capital (float): Starting capital for the backtest
            
        This method will:
        1. Apply entry/exit conditions to historical data
        2. Track all trades with detailed metrics
        3. Calculate performance statistics
        4. Update position tracking in real-time
        """
        if self.data is None:
            raise ValueError("No data loaded. Please fetch data first.")
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        
        # Add indicators
        self.add_basic_indicators()
        
        # Organize data by asset
        asset_data = {}
        for excel_name in self.data['excel_name'].unique():
            asset_df = self.data[self.data['excel_name'] == excel_name].copy()
            asset_data[excel_name] = asset_df
            print(f"\nAsset: {excel_name}")
            print(f"  Records: {len(asset_df)}")
            print(f"  Date range: {asset_df['Date'].min()} to {asset_df['Date'].max()}")
        
        # Extract strategy parameters
        entry_rules = strategy_rules['entry_conditions']
        exit_rules = strategy_rules['exit_conditions']
        position_size = strategy_rules['position_size']
        position_type = strategy_rules['position_type']
        
        # Find primary trading asset
        primary_asset_name = None
        for excel_name in asset_data.keys():
            if entry_rules['primary_asset'].lower() in excel_name.lower():
                primary_asset_name = excel_name
                break
        
        if primary_asset_name is None:
            primary_asset_name = list(asset_data.keys())[-1]
            print(f"\nUsing '{primary_asset_name}' as primary asset")
        
        primary_data = asset_data[primary_asset_name]
        
        # Trading loop
        position = None
        entry_signal_date = None
        
        for i in range(20, len(primary_data)):
            row = primary_data.iloc[i]
            current_date = row['Date']
            
            # Check entry conditions
            if position is None:
                signal_triggered = self._check_entry_signal(
                    entry_rules, current_date, row, asset_data, i, primary_data
                )
                
                if signal_triggered and entry_signal_date is None:
                    entry_signal_date = row['Date']
                
                if entry_signal_date is not None:
                    days_since_signal = (row['Date'] - entry_signal_date).days
                    
                    if days_since_signal >= entry_rules.get('delay_days', 0):
                        # Enter position
                        position = self._enter_position(
                            position_type, row, position_size, entry_signal_date
                        )
                        entry_signal_date = None
            
            # Check exit conditions
            else:
                position['bars_in_trade'] += 1
                position['highest_price'] = max(position['highest_price'], row['Price'])
                position['lowest_price'] = min(position['lowest_price'], row['Price'])
                
                if self._check_exit_conditions(exit_rules, position, row):
                    self._exit_position(position, row, primary_asset_name)
                    position = None
        
        # Calculate final metrics
        self._calculate_performance_metrics()
        
        # Print summary
        print(f"\nStrategy Execution Summary:")
        print(f"- Total trades executed: {len(self.trades)}")
        if len(self.trades) > 0:
            print(f"- First trade: {self.trades[0]['entry_date']}")
            print(f"- Last trade: {self.trades[-1]['exit_date']}")
    
    def _check_entry_signal(self, entry_rules, current_date, row, asset_data, i, primary_data):
        """Check if entry conditions are met."""
        signal_triggered = False
        
        # Multi-asset conditions
        if ',' in str(entry_rules.get('trigger_asset', '')):
            all_conditions_met = True
            
            for trigger_asset in entry_rules['trigger_asset'].split(','):
                trigger_asset = trigger_asset.strip()
                trigger_data = None
                
                for excel_name, df in asset_data.items():
                    if trigger_asset.lower() in excel_name.lower():
                        trigger_data = df
                        break
                
                if trigger_data is not None:
                    date_data = trigger_data[trigger_data['Date'] == current_date]
                    if not date_data.empty:
                        trigger_row = date_data.iloc[0]
                        
                        if pd.isna(trigger_row['Return']):
                            all_conditions_met = False
                            break
                        
                        if entry_rules['condition'] == 'rises':
                            condition_met = trigger_row['Return'] > 0
                        elif entry_rules['condition'] == 'drops':
                            condition_met = trigger_row['Return'] < 0
                        else:
                            condition_met = True
                        
                        if not condition_met:
                            all_conditions_met = False
                            break
                    else:
                        all_conditions_met = False
                        break
                else:
                    all_conditions_met = False
                    break
            
            signal_triggered = all_conditions_met
        
        # Single asset conditions
        else:
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
        
        return signal_triggered
    
    def _enter_position(self, position_type, row, position_size, entry_signal_date):
        """Enter a new position."""
        position = {
            'type': position_type,
            'entry_date': row['Date'],
            'entry_price': row['Price'],
            'quantity': (self.capital * position_size) / row['Price'],
            'signal_date': entry_signal_date,
            'bars_in_trade': 0,
            'highest_price': row['Price'],
            'lowest_price': row['Price'],
            'signal_name': f"CEMD_{position_type.title()}"
        }
        
        # Update max contracts tracking
        self.max_contracts_held = max(self.max_contracts_held, position['quantity'])
        if position_type == 'long':
            self.max_contracts_held_long = max(self.max_contracts_held_long, position['quantity'])
        else:
            self.max_contracts_held_short = max(self.max_contracts_held_short, position['quantity'])
        
        # Enhanced logging
        position_value = position['quantity'] * row['Price']
        print(f"\nEntering {position_type} position on {row['Date'].strftime('%Y-%m-%d')}:")
        print(f"  Entry Price: ${row['Price']:.2f}")
        print(f"  Position Size: {position['quantity']:.6f} units")
        print(f"  Position Value: ${position_value:,.2f}")
        print(f"  Capital Used: {position_size * 100:.1f}% (${position_value:,.2f})")
        
        return position
    
    def _check_exit_conditions(self, exit_rules, position, row):
        """Check if exit conditions are met."""
        exit_trade = False
        
        if exit_rules['type'] == 'time_based':
            days_held = (row['Date'] - position['entry_date']).days
            exit_trade = days_held >= exit_rules['value']
        
        elif exit_rules['type'] == 'profit_target':
            if position['type'] == 'long':
                profit_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
            else:
                profit_pct = ((position['entry_price'] - row['Price']) / position['entry_price']) * 100
            exit_trade = profit_pct >= exit_rules['value']
        
        elif exit_rules['type'] == 'stop_loss':
            if position['type'] == 'long':
                loss_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
            else:
                loss_pct = ((position['entry_price'] - row['Price']) / position['entry_price']) * 100
            exit_trade = loss_pct <= -exit_rules['value']
        
        return exit_trade
    
    def _exit_position(self, position, row, primary_asset_name):
        """Exit current position and record trade."""
        exit_price = row['Price']
        
        # Calculate P&L and metrics
        if position['type'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            run_up = (position['highest_price'] - position['entry_price']) * position['quantity']
            run_up_percent = ((position['highest_price'] - position['entry_price']) / position['entry_price']) * 100
            drawdown = (position['entry_price'] - position['lowest_price']) * position['quantity']
            drawdown_percent = ((position['entry_price'] - position['lowest_price']) / position['entry_price']) * 100
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            run_up = (position['entry_price'] - position['lowest_price']) * position['quantity']
            run_up_percent = ((position['entry_price'] - position['lowest_price']) / position['entry_price']) * 100
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
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            self.performance_metrics = self._get_empty_metrics()
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Long/Short breakdown
        long_trades = trades_df[trades_df['type'] == 'long']
        short_trades = trades_df[trades_df['type'] == 'short']
        
        # P&L metrics
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        net_profit = gross_profit - gross_loss
        
        # Commission
        commission_rate = self.strategy_properties['commission_rate']
        total_commission = sum([trade['entry_price'] * trade['quantity'] * commission_rate * 2 
                               for trade in self.trades])
        
        # Win/Loss ratios
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        returns = trades_df['pnl_percent'] / 100
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(trades_df['cumulative_pnl'])
        
        # Store all metrics
        self.performance_metrics = {
            'net_profit': net_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'commission_paid': total_commission,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'percent_profitable': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_equity_drawdown': max_drawdown,
            'max_contracts_held': self.max_contracts_held,
            'max_contracts_held_long': self.max_contracts_held_long,
            'max_contracts_held_short': self.max_contracts_held_short,
            'margin_calls': 0,
            'buy_hold_return': self._calculate_buy_hold_return(),
            'max_equity_runup': trades_df['cumulative_pnl'].max() if len(trades_df) > 0 else 0,
            # Additional metrics...
        }
        
        # Add all other metrics from the original implementation
        self._add_detailed_metrics(trades_df, long_trades, short_trades)
    
    def _get_empty_metrics(self):
        """Return empty metrics dictionary when no trades executed."""
        return {
            'total_trades': 0, 'net_profit': 0, 'winning_trades': 0,
            'losing_trades': 0, 'percent_profitable': 0, 'gross_profit': 0,
            'gross_loss': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'max_equity_drawdown': 0, 'commission_paid': 0,
            'max_contracts_held': 0, 'margin_calls': 0
        }
    
    def _add_detailed_metrics(self, trades_df, long_trades, short_trades):
        """Add detailed performance metrics to the metrics dictionary."""
        # Average metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        
        # Largest trades
        largest_win = trades_df['pnl'].max() if len(trades_df) > 0 else 0
        largest_loss = trades_df['pnl'].min() if len(trades_df) > 0 else 0
        
        # Duration metrics
        trades_df['duration'] = pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])
        avg_trade_duration = trades_df['duration'].mean()
        
        # Update metrics
        self.performance_metrics.update({
            'avg_winning_trade': avg_win,
            'avg_losing_trade': avg_loss,
            'ratio_avg_win_avg_loss': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'largest_winning_trade': largest_win,
            'largest_losing_trade': largest_loss,
            'largest_winning_trade_percent': trades_df['pnl_percent'].max() if len(trades_df) > 0 else 0,
            'largest_losing_trade_percent': trades_df['pnl_percent'].min() if len(trades_df) > 0 else 0,
            'avg_trade_duration': avg_trade_duration.days if pd.notna(avg_trade_duration) else 0,
            'avg_bars_in_trades': trades_df['bars_in_trade'].mean() if 'bars_in_trade' in trades_df.columns else 0,
            'avg_run_up': trades_df['run_up'].mean() if 'run_up' in trades_df.columns else 0,
            'avg_run_up_percent': trades_df['run_up_percent'].mean() if 'run_up_percent' in trades_df.columns else 0,
            'avg_drawdown': trades_df['drawdown'].mean() if 'drawdown' in trades_df.columns else 0,
            'avg_drawdown_percent': trades_df['drawdown_percent'].mean() if 'drawdown_percent' in trades_df.columns else 0,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_percent_profitable': (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
            'short_percent_profitable': (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0,
        })
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio."""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_pnl) == 0:
            return 0
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + self.initial_capital)
        return abs(drawdown.min() * 100)
    
    def _calculate_buy_hold_return(self) -> float:
        """Calculate buy and hold return."""
        if self.data is None or len(self.data) == 0:
            return 0
        
        bitcoin_data = self.data[self.data['excel_name'].str.contains('Bitcoin', case=False)]
        if len(bitcoin_data) == 0:
            return 0
            
        first_price = bitcoin_data.iloc[0]['Close']
        last_price = bitcoin_data.iloc[-1]['Close']
        return ((last_price - first_price) / first_price) * self.initial_capital
    
    def generate_report(self, save_path: str = 'backtest_report.html'):
        """Generate comprehensive HTML report with all metrics and visualizations."""
        html_content = self._generate_html_report()
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {save_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML content for the report."""
        # Implementation details omitted for brevity
        # This would include the full HTML template with metrics display
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cryptocurrency Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1a1a1a; color: #e0e0e0; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background-color: #2a2a2a; border-radius: 8px; padding: 20px; }}
                .positive {{ color: #4caf50; }}
                .negative {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cryptocurrency Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Net Profit</h3>
                        <p class="{'positive' if self.performance_metrics.get('net_profit', 0) >= 0 else 'negative'}">
                            ${self.performance_metrics.get('net_profit', 0):,.2f}
                        </p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Trades</h3>
                        <p>{self.performance_metrics.get('total_trades', 0)}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Win Rate</h3>
                        <p>{self.performance_metrics.get('percent_profitable', 0):.1f}%</p>
                    </div>
                    <div class="metric-card">
                        <h3>Sharpe Ratio</h3>
                        <p>{self.performance_metrics.get('sharpe_ratio', 0):.3f}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def plot_equity_curve(self):
        """Plot interactive equity curve using Plotly."""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            trades_df = pd.DataFrame(self.trades)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades_df['exit_date'],
                y=trades_df['cumulative_pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(color=self.colors['green'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 215, 117, 0.1)'
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title=f'Equity Curve - Total Return: ${self.performance_metrics["net_profit"]:,.2f}',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L ($)',
                template='plotly_dark',
                height=600
            )
            
            output_file = 'equity_curve_interactive.html'
            fig.write_html(output_file)
            print(f"Interactive equity curve saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating equity curve: {e}")
            self._plot_simple_equity_curve()
    
    def _plot_simple_equity_curve(self):
        """Fallback simple matplotlib equity curve."""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            trades_df = pd.DataFrame(self.trades)
            ax.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], 
                   color=self.colors['green'], linewidth=2)
            ax.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                           color=self.colors['green'], alpha=0.3)
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            
            ax.set_title(f'Equity Curve - Total Return: ${self.performance_metrics["net_profit"]:,.2f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative P&L ($)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('equity_curve_simple.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Simple equity curve saved to equity_curve_simple.png")
            
        except Exception as e:
            print(f"Fallback plot also failed: {e}")
    
    def plot_trade_distribution(self):
        """Plot trade distribution analysis."""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            trades_df = pd.DataFrame(self.trades)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('P&L Distribution', 'Win Rate Evolution', 
                               'Monthly Returns', 'Trade P&L')
            )
            
            # P&L Distribution
            fig.add_trace(
                go.Histogram(x=trades_df['pnl'], nbinsx=30, name='P&L'),
                row=1, col=1
            )
            
            # Win Rate Evolution
            trades_df['cumulative_win_rate'] = (trades_df['pnl'] > 0).expanding().mean() * 100
            fig.add_trace(
                go.Scatter(x=trades_df['trade_id'], y=trades_df['cumulative_win_rate'], 
                          mode='lines', name='Win Rate'),
                row=1, col=2
            )
            
            # Monthly Returns
            trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_returns = trades_df.groupby('month')['pnl'].sum()
            
            fig.add_trace(
                go.Bar(x=[str(m) for m in monthly_returns.index], 
                      y=monthly_returns.values, name='Monthly P&L'),
                row=2, col=1
            )
            
            # Individual Trade P&L
            colors = [self.colors['green'] if x > 0 else self.colors['red'] for x in trades_df['pnl']]
            fig.add_trace(
                go.Bar(x=trades_df['trade_id'], y=trades_df['pnl'], 
                      marker_color=colors, name='Trade P&L'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text='Trade Distribution Analysis',
                showlegend=False,
                template='plotly_dark',
                height=800
            )
            
            output_file = 'trade_distribution_interactive.html'
            fig.write_html(output_file)
            print(f"Interactive trade distribution saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating trade distribution: {e}")
    
    def plot_comprehensive_dashboard(self):
        """Create a comprehensive trading dashboard."""
        if not self.trades:
            print("No trades to plot")
            return
        
        try:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(16, 10))
            
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            trades_df = pd.DataFrame(self.trades)
            
            # Equity Curve
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], 
                    color=self.colors['green'], linewidth=2)
            ax1.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                           color=self.colors['green'], alpha=0.3)
            ax1.set_title('Equity Curve', fontsize=14)
            ax1.set_ylabel('Cumulative P&L ($)')
            ax1.grid(True, alpha=0.3)
            
            # Key Metrics
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            metrics_text = f"""Key Metrics:
            
Net Profit: ${self.performance_metrics.get('net_profit', 0):,.2f}
Win Rate: {self.performance_metrics.get('percent_profitable', 0):.1f}%
Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {self.performance_metrics.get('max_equity_drawdown', 0):.1f}%
Total Trades: {self.performance_metrics.get('total_trades', 0)}"""
            
            ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            plt.suptitle('Trading Performance Dashboard', fontsize=16)
            plt.tight_layout()
            plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Comprehensive dashboard saved to comprehensive_dashboard.png")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
    
    def print_summary(self):
        """Print performance summary to console."""
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        metrics = self.performance_metrics
        
        print(f"\nOverview:")
        print(f"  Net Profit: ${metrics.get('net_profit', 0):,.2f}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('percent_profitable', 0):.1f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_equity_drawdown', 0):.2f}%")
        
        print("\n" + "="*60)