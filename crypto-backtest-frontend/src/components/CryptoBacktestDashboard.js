import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { 
  TrendingUp, TrendingDown, Play, Pause, RefreshCw, Download,
  DollarSign, Percent, Activity, BarChart3, PieChart, 
  Settings, Info, AlertCircle, CheckCircle, ChevronDown, ChevronUp,
  Moon, Sun, Code, Brain, Database, Zap, ArrowUpRight, ArrowDownRight,
  Calendar, Clock, Target, Shield, Layers, Eye, EyeOff, X,
  FileText, Copy, ExternalLink, HelpCircle, AlertTriangle,
  Loader2, ChevronRight, Filter, Search, Menu, Save
} from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  PieChart as RePieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Scatter, RadarChart, PolarGrid, 
  PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Constants
const STRATEGY_EXAMPLES = [
  "When SPY goes down buy BTC 5 days later",
  "Buy ETH when it drops 10% and sell after 7 days",
  "Short BTC when price crosses below 20-day average, exit with 5% profit target",
  "When DXY up AND 30-Year TIPS up, BUY Bitcoin 4 days later"
];

const DATA_SOURCES = [
  { id: 'btc', name: 'Bitcoin Daily Close Price', category: 'crypto' },
  { id: 'eth', name: 'Ethereum Daily Close Price', category: 'crypto' },
  { id: 'dxy', name: 'DXY Daily Close Price', category: 'forex' },
  { id: 'tips', name: '30-Year TIPS Yield (%)', category: 'bonds' },
  { id: 'spy', name: 'SPY Daily Close Price', category: 'stocks' }
];

const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  danger: '#ef4444',
  warning: '#f59e0b',
  info: '#06b6d4'
};

// Custom hooks
const useLocalStorage = (key, defaultValue) => {
  const [value, setValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error);
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error);
    }
  }, [key, value]);

  return [value, setValue];
};

const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
};

// Components
const ErrorBoundary = ({ children }) => {
  const [hasError, setHasError] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const errorHandler = (error) => {
      setHasError(true);
      setError(error);
    };

    window.addEventListener('error', errorHandler);
    return () => window.removeEventListener('error', errorHandler);
  }, []);

  if (hasError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center p-8">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-white mb-2">Something went wrong</h1>
          <p className="text-gray-400 mb-4">An error occurred while loading the application</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return children;
};

const LoadingSpinner = ({ size = 'md' }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <div className="flex items-center justify-center p-8">
      <Loader2 className={`${sizes[size]} animate-spin text-blue-500`} />
    </div>
  );
};

const Alert = ({ type = 'info', title, children, onClose, action }) => {
  const styles = {
    info: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    success: 'bg-green-500/10 border-green-500/30 text-green-400',
    warning: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400',
    error: 'bg-red-500/10 border-red-500/30 text-red-400'
  };

  const icons = {
    info: Info,
    success: CheckCircle,
    warning: AlertTriangle,
    error: AlertCircle
  };

  const Icon = icons[type];

  return (
    <div className={`relative p-4 rounded-lg border ${styles[type]}`}>
      <div className="flex items-start">
        <Icon className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          {title && <h4 className="font-semibold mb-1">{title}</h4>}
          <div className="text-sm">{children}</div>
          {action && (
            <div className="mt-3">
              {action}
            </div>
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-3 -mr-1 -mt-1 p-1 hover:bg-white/10 rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

const Modal = ({ isOpen, onClose, title, children, size = 'md' }) => {
  const sizeClasses = {
    sm: 'max-w-md',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
    xl: 'max-w-6xl'
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4">
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
        
        <div className={`relative bg-gray-800 rounded-2xl shadow-2xl w-full ${sizeClasses[size]} max-h-[90vh] overflow-hidden`}>
          <div className="flex items-center justify-between p-6 border-b border-gray-700">
            <h3 className="text-xl font-semibold text-white">{title}</h3>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

const Tabs = ({ tabs, activeTab, onChange }) => {
  return (
    <div className="flex space-x-1 p-1 bg-gray-800/50 rounded-lg">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex-1 flex items-center justify-center px-4 py-2 rounded-md font-medium transition-all
            ${activeTab === tab.id 
              ? 'bg-gray-700 text-white shadow-sm' 
              : 'text-gray-400 hover:text-white hover:bg-gray-700/50'}
          `}
        >
          {tab.icon && <tab.icon className="w-4 h-4 mr-2" />}
          {tab.label}
        </button>
      ))}
    </div>
  );
};

const MetricCard = ({ icon: Icon, title, value, change, trend, color = 'blue', onClick }) => {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/20 border-green-500/30',
    red: 'from-red-500/20 to-red-600/20 border-red-500/30',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30',
    yellow: 'from-yellow-500/20 to-yellow-600/20 border-yellow-500/30'
  };

  return (
    <div
      onClick={onClick}
      className={`
        relative p-6 rounded-2xl border backdrop-blur-xl
        bg-gradient-to-br ${colorClasses[color]}
        transition-all duration-300 hover:scale-105 hover:shadow-2xl
        ${onClick ? 'cursor-pointer' : ''}
      `}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-white mb-2">{value}</p>
          {(change !== undefined || trend) && (
            <div className="flex items-center space-x-2">
              {change !== undefined && (
                <span className={`text-sm flex items-center ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {change >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                  {Math.abs(change).toFixed(1)}%
                </span>
              )}
              {trend && <span className="text-xs text-gray-500">{trend}</span>}
            </div>
          )}
        </div>
        <Icon className={`w-8 h-8 text-${color}-400`} />
      </div>
    </div>
  );
};

// Main Component
const CryptoBacktestDashboard = () => {
  // Theme
  const [theme, setTheme] = useLocalStorage('theme', 'dark');
  const isDark = theme === 'dark';

  // Navigation
  const [activeView, setActiveView] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Form State
  const [formData, setFormData] = useState({
    strategy: '',
    dataSources: [],
    initialCapital: '100000',
    pineconeKey: '',
    openaiKey: '',
    positionSize: '10',
    commissionRate: '0.1',
    slippage: '0',
    maxPositions: '1'
  });

  // API Keys visibility
  const [showApiKeys, setShowApiKeys] = useState(false);

  // Backtest State
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('idle');
  const [results, setResults] = useState(null);
  const [parsedStrategy, setParsedStrategy] = useState(null);
  const [error, setError] = useState(null);

  // UI State
  const [selectedTimeframe, setSelectedTimeframe] = useState('1M');
  const [activeTab, setActiveTab] = useState('overview');
  const [showHelp, setShowHelp] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  // Refs
  const strategyInputRef = useRef(null);

  // Debounced values
  const debouncedStrategy = useDebounce(formData.strategy, 500);

  // Computed values
  const isFormValid = useMemo(() => {
    return formData.strategy.trim() !== '' && 
           formData.dataSources.length > 0 &&
           parseInt(formData.initialCapital) > 0;
  }, [formData]);

  const canRunBacktest = useMemo(() => {
    return isFormValid && parsedStrategy && !isRunning;
  }, [isFormValid, parsedStrategy, isRunning]);

  // Handlers
  const handleFormChange = useCallback((field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  const handleDataSourceToggle = useCallback((sourceId) => {
    setFormData(prev => ({
      ...prev,
      dataSources: prev.dataSources.includes(sourceId)
        ? prev.dataSources.filter(id => id !== sourceId)
        : [...prev.dataSources, sourceId]
    }));
  }, []);

  const handleParseStrategy = useCallback(async () => {
    if (!formData.strategy.trim()) return;

    setError(null);
    setIsRunning(true);
    setCurrentStep('parsing');
    setProgress(0);

    try {
      const res = await fetch(`${API_URL}/api/parse-strategy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy: formData.strategy })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API error');
      setParsedStrategy(data.strategy_rules);
      setCurrentStep('ready');
    } catch (err) {
      setError('Failed to parse strategy. ' + err.message);
      setCurrentStep('idle');
    } finally {
      setIsRunning(false);
      setProgress(0);
    }
  }, [formData.strategy]);

  const handleRunBacktest = useCallback(async () => {
    if (!canRunBacktest) return;

    setError(null);
    setIsRunning(true);
    setCurrentStep('backtesting');
    setProgress(0);

    try {
      const res = await fetch(`${API_URL}/api/run-backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_rules: parsedStrategy,
          excel_names: formData.dataSources,
          initial_capital: parseFloat(formData.initialCapital)
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API error');
      setResults(data.performance);
      setCurrentStep('complete');
      setActiveView('results');
    } catch (err) {
      setError('Backtest failed. ' + err.message);
      setCurrentStep('ready');
    } finally {
      setIsRunning(false);
    }
  }, [canRunBacktest, parsedStrategy, formData.dataSources, formData.initialCapital]);

  const handleReset = useCallback(() => {
    setFormData({
      strategy: '',
      dataSources: [],
      initialCapital: '100000',
      pineconeKey: '',
      openaiKey: '',
      positionSize: '10',
      commissionRate: '0.1',
      slippage: '0',
      maxPositions: '1'
    });
    setParsedStrategy(null);
    setResults(null);
    setCurrentStep('idle');
    setProgress(0);
    setError(null);
    setActiveView('dashboard');
  }, []);

  const handleExport = useCallback((format) => {
    // Simulate export functionality
    console.log(`Exporting results in ${format} format...`);
    setShowExportModal(false);
  }, []);

  // Generate chart data
  const generateChartData = useMemo(() => {
    if (!results) return null;

    const equityData = [];
    let equity = parseInt(formData.initialCapital);
    const days = 252; // One trading year

    for (let i = 0; i < days; i++) {
      const dailyReturn = (Math.random() - 0.48) * 0.02;
      equity = equity * (1 + dailyReturn);
      equityData.push({
        day: i + 1,
        date: new Date(2024, 0, i + 1).toLocaleDateString(),
        equity: Math.round(equity),
        drawdown: equity < parseInt(formData.initialCapital) 
          ? ((equity - parseInt(formData.initialCapital)) / parseInt(formData.initialCapital)) * 100 
          : 0,
        benchmark: parseInt(formData.initialCapital) + (i * 100)
      });
    }

    const monthlyReturns = [];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    for (let i = 0; i < 12; i++) {
      const monthlyReturn = (Math.random() - 0.45) * 10000;
      monthlyReturns.push({
        month: months[i],
        returns: monthlyReturn,
        positive: monthlyReturn > 0
      });
    }

    const tradeDistribution = [
      { name: 'Winning Trades', value: results.winningTrades, fill: CHART_COLORS.success },
      { name: 'Losing Trades', value: results.losingTrades, fill: CHART_COLORS.danger }
    ];

    const performanceRadar = [
      { metric: 'Win Rate', value: results.winRate, fullMark: 100 },
      { metric: 'Profit Factor', value: Math.min(results.profitFactor * 20, 100), fullMark: 100 },
      { metric: 'Sharpe Ratio', value: Math.min(results.sharpeRatio * 30, 100), fullMark: 100 },
      { metric: 'Recovery', value: 100 - Math.abs(results.maxDrawdown), fullMark: 100 },
      { metric: 'Consistency', value: 60 + Math.random() * 30, fullMark: 100 }
    ];

    return { equityData, monthlyReturns, tradeDistribution, performanceRadar };
  }, [results, formData.initialCapital]);

  // Render functions
  const renderStrategyForm = () => (
    <div className="space-y-6">
      {/* Strategy Input */}
      <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-white flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-400" />
            Trading Strategy
          </h3>
          <button
            onClick={() => setShowHelp(true)}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <HelpCircle className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        <div className="space-y-4">
          <div className="relative">
            <textarea
              ref={strategyInputRef}
              value={formData.strategy}
              onChange={(e) => handleFormChange('strategy', e.target.value)}
              placeholder="Describe your trading strategy in plain English..."
              className="w-full h-32 bg-gray-900/50 border border-gray-700 rounded-xl p-4 
                       text-white placeholder-gray-500 resize-none
                       focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 focus:outline-none
                       transition-all duration-200"
            />
            <div className="absolute bottom-2 right-2 text-xs text-gray-500">
              {formData.strategy.length} characters
            </div>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-3">Quick Examples:</p>
            <div className="flex flex-wrap gap-2">
              {STRATEGY_EXAMPLES.map((example, i) => (
                <button
                  key={i}
                  onClick={() => handleFormChange('strategy', example)}
                  className="px-3 py-1.5 bg-gray-700/50 hover:bg-gray-600/50 
                           border border-gray-600 rounded-lg text-sm text-gray-300
                           transition-all duration-200 hover:scale-105"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {debouncedStrategy && !parsedStrategy && (
            <Alert type="info" title="Ready to parse">
              Click "Parse Strategy" to convert your strategy into trading rules
            </Alert>
          )}
        </div>
      </div>

      {/* Data Sources */}
      <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Database className="w-5 h-5 mr-2 text-blue-400" />
          Data Sources
        </h3>

        <div className="space-y-3">
          {Object.entries(
            DATA_SOURCES.reduce((acc, source) => {
              if (!acc[source.category]) acc[source.category] = [];
              acc[source.category].push(source);
              return acc;
            }, {})
          ).map(([category, sources]) => (
            <div key={category}>
              <p className="text-sm text-gray-400 mb-2 capitalize">{category}</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sources.map((source) => (
                  <label
                    key={source.id}
                    className={`
                      flex items-center p-3 rounded-lg border cursor-pointer
                      transition-all duration-200 hover:scale-[1.02]
                      ${formData.dataSources.includes(source.id)
                        ? 'bg-blue-500/20 border-blue-500/50 text-white'
                        : 'bg-gray-900/50 border-gray-700 text-gray-300 hover:border-gray-600'}
                    `}
                  >
                    <input
                      type="checkbox"
                      checked={formData.dataSources.includes(source.id)}
                      onChange={() => handleDataSourceToggle(source.id)}
                      className="sr-only"
                    />
                    <div className={`
                      w-5 h-5 rounded border-2 mr-3 flex items-center justify-center
                      transition-all duration-200
                      ${formData.dataSources.includes(source.id)
                        ? 'bg-blue-500 border-blue-500'
                        : 'border-gray-600'}
                    `}>
                      {formData.dataSources.includes(source.id) && (
                        <CheckCircle className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <span className="text-sm">{source.name}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>

        {formData.dataSources.length === 0 && (
          <Alert type="warning" title="No data sources selected">
            Please select at least one data source for your backtest
          </Alert>
        )}
      </div>

      {/* Configuration */}
      <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-green-400" />
          Configuration
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Initial Capital
            </label>
            <div className="relative">
              <DollarSign className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
              <input
                type="number"
                value={formData.initialCapital}
                onChange={(e) => handleFormChange('initialCapital', e.target.value)}
                className="w-full bg-gray-900/50 border border-gray-700 rounded-xl pl-10 pr-4 py-3
                         text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 
                         focus:outline-none transition-all duration-200"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Position Size (%)
            </label>
            <div className="relative">
              <Percent className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
              <input
                type="number"
                value={formData.positionSize}
                onChange={(e) => handleFormChange('positionSize', e.target.value)}
                min="1"
                max="100"
                className="w-full bg-gray-900/50 border border-gray-700 rounded-xl pl-10 pr-4 py-3
                         text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 
                         focus:outline-none transition-all duration-200"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Commission Rate (%)
            </label>
            <input
              type="number"
              value={formData.commissionRate}
              onChange={(e) => handleFormChange('commissionRate', e.target.value)}
              step="0.01"
              className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-3
                       text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 
                       focus:outline-none transition-all duration-200"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Max Positions
            </label>
            <input
              type="number"
              value={formData.maxPositions}
              onChange={(e) => handleFormChange('maxPositions', e.target.value)}
              min="1"
              className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-3
                       text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 
                       focus:outline-none transition-all duration-200"
            />
          </div>
        </div>

        {/* API Keys Section */}
        <div className="mt-6 pt-6 border-t border-gray-700">
          <button
            onClick={() => setShowApiKeys(!showApiKeys)}
            className="flex items-center text-sm text-gray-400 hover:text-white transition-colors"
          >
            <Shield className="w-4 h-4 mr-2" />
            API Configuration
            {showApiKeys ? <ChevronUp className="w-4 h-4 ml-2" /> : <ChevronDown className="w-4 h-4 ml-2" />}
          </button>

          {showApiKeys && (
            <div className="mt-4 space-y-4">
              <Alert type="info">
                Your API keys are stored locally and never sent to our servers
              </Alert>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Pinecone API Key
                </label>
                <div className="relative">
                  <input
                    type="password"
                    value={formData.pineconeKey}
                    onChange={(e) => handleFormChange('pineconeKey', e.target.value)}
                    placeholder="pcsk_..."
                    className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-3 pr-10
                             text-white placeholder-gray-500 focus:border-blue-500 focus:ring-2 
                             focus:ring-blue-500/20 focus:outline-none transition-all duration-200"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-3 text-gray-500 hover:text-gray-300"
                  >
                    <Eye className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  OpenAI API Key
                </label>
                <div className="relative">
                  <input
                    type="password"
                    value={formData.openaiKey}
                    onChange={(e) => handleFormChange('openaiKey', e.target.value)}
                    placeholder="sk-..."
                    className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-3 pr-10
                             text-white placeholder-gray-500 focus:border-blue-500 focus:ring-2 
                             focus:ring-blue-500/20 focus:outline-none transition-all duration-200"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-3 text-gray-500 hover:text-gray-300"
                  >
                    <Eye className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between items-center">
        <button
          onClick={handleReset}
          className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
        >
          Reset Form
        </button>

        <div className="flex gap-3">
          {parsedStrategy && (
            <button
              onClick={handleRunBacktest}
              disabled={!canRunBacktest}
              className={`
                px-6 py-3 rounded-xl font-semibold flex items-center gap-2
                transition-all duration-200 hover:scale-105
                ${canRunBacktest
                  ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:shadow-lg hover:shadow-green-500/25'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'}
              `}
            >
              <Play className="w-5 h-5" />
              Run Backtest
            </button>
          )}

          <button
            onClick={handleParseStrategy}
            disabled={!isFormValid || isRunning}
            className={`
              px-6 py-3 rounded-xl font-semibold flex items-center gap-2
              transition-all duration-200 hover:scale-105
              ${isFormValid && !isRunning
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:shadow-lg hover:shadow-blue-500/25'
                : 'bg-gray-700 text-gray-400 cursor-not-allowed'}
            `}
          >
            {isRunning && currentStep === 'parsing' ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Parsing...
              </>
            ) : (
              <>
                <Brain className="w-5 h-5" />
                Parse Strategy
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );

  const renderParsedStrategy = () => (
    <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-white flex items-center">
          <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
          Strategy Parsed Successfully
        </h3>
        <button
          onClick={() => setParsedStrategy(null)}
          className="text-gray-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Position Type</p>
          <p className="font-medium text-white uppercase">{parsedStrategy.positionType}</p>
        </div>
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Entry Trigger</p>
          <p className="font-medium text-white">
            {parsedStrategy.entryConditions.triggerAsset} {parsedStrategy.entryConditions.condition}
          </p>
        </div>
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Entry Delay</p>
          <p className="font-medium text-white">{parsedStrategy.entryConditions.delayDays} days</p>
        </div>
        <div className="p-4 bg-gray-900/50 rounded-lg">
          <p className="text-sm text-gray-400 mb-1">Exit Strategy</p>
          <p className="font-medium text-white">
            {parsedStrategy.exitConditions.type.replace('_', ' ')} - {parsedStrategy.exitConditions.value} days
          </p>
        </div>
      </div>
    </div>
  );

  const renderResults = () => {
    if (!results || !generateChartData) return null;

    const { equityData, monthlyReturns, tradeDistribution, performanceRadar } = generateChartData;

    const tabs = [
      { id: 'overview', label: 'Overview', icon: BarChart3 },
      { id: 'performance', label: 'Performance', icon: TrendingUp },
      { id: 'risk', label: 'Risk Analysis', icon: Shield },
      { id: 'trades', label: 'Trade History', icon: Activity }
    ];

    return (
      <div className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            icon={DollarSign}
            title="Net Profit"
            value={`$${results.netProfit.toLocaleString()}`}
            change={((results.netProfit / parseInt(formData.initialCapital)) * 100)}
            color={results.netProfit > 0 ? 'green' : 'red'}
          />
          <MetricCard
            icon={Percent}
            title="Win Rate"
            value={`${results.winRate.toFixed(1)}%`}
            trend="Last 30 days"
            color="blue"
          />
          <MetricCard
            icon={Activity}
            title="Sharpe Ratio"
            value={results.sharpeRatio.toFixed(2)}
            color="purple"
          />
          <MetricCard
            icon={TrendingDown}
            title="Max Drawdown"
            value={`${results.maxDrawdown.toFixed(1)}%`}
            color="red"
          />
        </div>

        {/* Main Content */}
        <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl border border-gray-700">
          <div className="p-6">
            <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
          </div>

          <div className="p-6 pt-0">
            {activeTab === 'overview' && (
              <div className="space-y-8">
                {/* Equity Curve */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white">Equity Curve</h4>
                    <div className="flex gap-2">
                      {['1M', '3M', '6M', '1Y', 'ALL'].map((tf) => (
                        <button
                          key={tf}
                          onClick={() => setSelectedTimeframe(tf)}
                          className={`
                            px-3 py-1 rounded-lg text-sm font-medium transition-colors
                            ${selectedTimeframe === tf
                              ? 'bg-blue-500 text-white'
                              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}
                          `}
                        >
                          {tf}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="h-80 -mx-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityData}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.3}/>
                            <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="day" 
                          stroke="#9ca3af"
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis 
                          stroke="#9ca3af"
                          tick={{ fontSize: 12 }}
                          domain={['dataMin - 5000', 'dataMax + 5000']}
                        />
                        <Tooltip
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px'
                          }}
                          labelFormatter={(value) => `Day ${value}`}
                        />
                        <Area
                          type="monotone"
                          dataKey="equity"
                          stroke={CHART_COLORS.primary}
                          strokeWidth={2}
                          fillOpacity={1}
                          fill="url(#equityGradient)"
                        />
                        <Line
                          type="monotone"
                          dataKey="benchmark"
                          stroke="#6b7280"
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Distribution Charts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Win/Loss Distribution</h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <RePieChart>
                          <Pie
                            data={tradeDistribution}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {tradeDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{ 
                              backgroundColor: '#1f2937', 
                              border: '1px solid #374151',
                              borderRadius: '8px'
                            }}
                          />
                          <Legend />
                        </RePieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Total Trades</span>
                        <span className="text-white font-medium">{results.totalTrades}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Profit Factor</span>
                        <span className="text-white font-medium">{results.profitFactor.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Expectancy</span>
                        <span className="text-white font-medium">${results.expectancy.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Monthly Returns</h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={monthlyReturns}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="month" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                          <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
                          <Tooltip
                            contentStyle={{ 
                              backgroundColor: '#1f2937', 
                              border: '1px solid #374151',
                              borderRadius: '8px'
                            }}
                            formatter={(value) => `$${value.toFixed(2)}`}
                          />
                          <Bar dataKey="returns">
                            {monthlyReturns.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.positive ? CHART_COLORS.success : CHART_COLORS.danger} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'performance' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-white">Profit Metrics</h4>
                    <div className="space-y-3">
                      {[
                        ['Gross Profit', `$${results.grossProfit.toLocaleString()}`, 'text-green-400'],
                        ['Gross Loss', `$${Math.abs(results.grossLoss).toLocaleString()}`, 'text-red-400'],
                        ['Net Profit', `$${results.netProfit.toLocaleString()}`, results.netProfit > 0 ? 'text-green-400' : 'text-red-400'],
                        ['Commission Paid', `$${results.commission.toFixed(2)}`, 'text-gray-400'],
                        ['Average Win', `$${results.avgWin.toFixed(2)}`, 'text-green-400'],
                        ['Average Loss', `$${Math.abs(results.avgLoss).toFixed(2)}`, 'text-red-400'],
                        ['Largest Win', `$${results.largestWin.toFixed(2)}`, 'text-green-400'],
                        ['Largest Loss', `$${Math.abs(results.largestLoss).toFixed(2)}`, 'text-red-400']
                      ].map(([label, value, color]) => (
                        <div key={label} className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                          <span className="text-gray-400">{label}</span>
                          <span className={`font-medium ${color}`}>{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Performance Radar</h4>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={performanceRadar}>
                          <PolarGrid stroke="#374151" />
                          <PolarAngleAxis dataKey="metric" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                          <PolarRadiusAxis stroke="#9ca3af" tick={{ fontSize: 10 }} />
                          <Radar
                            dataKey="value"
                            stroke={CHART_COLORS.primary}
                            fill={CHART_COLORS.primary}
                            fillOpacity={0.3}
                          />
                          <Tooltip
                            contentStyle={{ 
                              backgroundColor: '#1f2937', 
                              border: '1px solid #374151',
                              borderRadius: '8px'
                            }}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'risk' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { label: 'Sharpe Ratio', value: results.sharpeRatio.toFixed(3), icon: Activity, color: 'blue' },
                    { label: 'Sortino Ratio', value: results.sortinoRatio.toFixed(3), icon: TrendingUp, color: 'green' },
                    { label: 'Max Drawdown', value: `${results.maxDrawdown.toFixed(2)}%`, icon: TrendingDown, color: 'red' },
                    { label: 'Max Contracts', value: results.maxContracts.toFixed(2), icon: Layers, color: 'purple' },
                    { label: 'Win Rate', value: `${results.winRate.toFixed(1)}%`, icon: Target, color: 'green' },
                    { label: 'Avg Duration', value: `${results.avgTradeDuration.toFixed(1)} days`, icon: Clock, color: 'blue' }
                  ].map(({ label, value, icon: Icon, color }) => (
                    <div key={label} className={`p-4 bg-gray-900/50 rounded-lg border border-gray-700`}>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-400 text-sm">{label}</span>
                        <Icon className={`w-5 h-5 text-${color}-400`} />
                      </div>
                      <p className="text-2xl font-bold text-white">{value}</p>
                    </div>
                  ))}
                </div>

                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">Drawdown Analysis</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="day" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                        <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
                        <Tooltip
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px'
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="drawdown"
                          stroke={CHART_COLORS.danger}
                          fill={CHART_COLORS.danger}
                          fillOpacity={0.3}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'trades' && (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Trade #</th>
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Type</th>
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Entry Date</th>
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Exit Date</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">Entry Price</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">Exit Price</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">P&L</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">P&L %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...Array(20)].map((_, i) => {
                      const isProfit = Math.random() > 0.43;
                      const pnl = isProfit ? Math.random() * 1000 + 100 : -(Math.random() * 600 + 100);
                      const pnlPercent = pnl / 10000 * 100;
                      
                      return (
                        <tr key={i} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                          <td className="py-3 px-4 text-white">{i + 1}</td>
                          <td className="py-3 px-4">
                            <span className={`
                              px-2 py-1 rounded text-xs font-medium
                              ${i % 3 === 0 
                                ? 'bg-red-500/20 text-red-400' 
                                : 'bg-green-500/20 text-green-400'}
                            `}>
                              {i % 3 === 0 ? 'SHORT' : 'LONG'}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-gray-300">2024-{String(Math.floor(i/30) + 1).padStart(2, '0')}-{String((i % 30) + 1).padStart(2, '0')}</td>
                          <td className="py-3 px-4 text-gray-300">2024-{String(Math.floor(i/30) + 1).padStart(2, '0')}-{String((i % 30) + 11).padStart(2, '0')}</td>
                          <td className="py-3 px-4 text-right text-gray-300">${(50000 + Math.random() * 20000).toFixed(2)}</td>
                          <td className="py-3 px-4 text-right text-gray-300">${(50000 + Math.random() * 20000).toFixed(2)}</td>
                          <td className={`py-3 px-4 text-right font-medium ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                            ${isProfit ? '+' : ''}{pnl.toFixed(2)}
                          </td>
                          <td className={`py-3 px-4 text-right font-medium ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                            {isProfit ? '+' : ''}{pnlPercent.toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Export Actions */}
          <div className="p-6 border-t border-gray-700 flex justify-between items-center">
            <div className="flex gap-2">
              <button
                onClick={() => setShowExportModal(true)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg 
                         text-white font-medium transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export Results
              </button>
              <button
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg 
                         text-white font-medium transition-colors flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                Save Strategy
              </button>
            </div>

            <button
              onClick={handleReset}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              New Backtest
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderProgress = () => {
    if (!isRunning) return null;

    const steps = {
      parsing: 'Parsing your strategy...',
      loading_data: 'Loading historical data...',
      calculating_signals: 'Calculating trading signals...',
      executing_trades: 'Executing trades...',
      calculating_metrics: 'Calculating performance metrics...',
      generating_reports: 'Generating reports...'
    };

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-gray-800 rounded-2xl p-8 max-w-md w-full mx-4">
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-4 relative">
              <div className="absolute inset-0 bg-blue-500/20 rounded-full animate-ping" />
              <div className="relative w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <Zap className="w-10 h-10 text-white" />
              </div>
            </div>

            <h3 className="text-xl font-semibold text-white mb-2">
              Running Backtest
            </h3>
            <p className="text-gray-400 mb-6">
              {steps[currentStep] || 'Processing...'}
            </p>

            <div className="w-full bg-gray-700 rounded-full h-2 mb-2 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-500">{progress}% complete</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <ErrorBoundary>
      <div className={`min-h-screen ${isDark ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
        {/* Background Effects */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000" />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-4000" />
        </div>

        {/* Header */}
        <header className="relative z-20 border-b border-gray-800 bg-gray-900/50 backdrop-blur-xl">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors lg:hidden"
                >
                  <Menu className="w-5 h-5" />
                </button>
                
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
                    CryptoBacktest Pro
                  </h1>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setTheme(isDark ? 'light' : 'dark')}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                  aria-label="Toggle theme"
                >
                  {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </button>

                <button
                  onClick={() => setShowHelp(true)}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                  aria-label="Help"
                >
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Layout */}
        <div className="relative z-10 flex">
          {/* Sidebar */}
          <aside className={`
            fixed lg:sticky top-0 left-0 h-screen w-64 bg-gray-800/50 backdrop-blur-xl
            border-r border-gray-700 transform transition-transform duration-300 z-30
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          `}>
            <nav className="p-4 space-y-2">
              {[
                { id: 'dashboard', label: 'Strategy Builder', icon: Brain },
                { id: 'results', label: 'Results', icon: BarChart3, disabled: !results },
                { id: 'history', label: 'History', icon: Clock },
                { id: 'settings', label: 'Settings', icon: Settings }
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    if (!item.disabled) {
                      setActiveView(item.id);
                      setSidebarOpen(false);
                    }
                  }}
                  disabled={item.disabled}
                  className={`
                    w-full flex items-center space-x-3 px-4 py-3 rounded-lg
                    transition-all duration-200
                    ${activeView === item.id
                      ? 'bg-gray-700 text-white'
                      : item.disabled
                      ? 'text-gray-500 cursor-not-allowed'
                      : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'}
                  `}
                >
                  <item.icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              ))}
            </nav>

            <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-700">
              <div className="text-xs text-gray-400">
                <p className="mb-1">Version 1.0.0</p>
                <p>© 2024 CryptoBacktest Pro</p>
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 min-h-screen">
            <div className="container mx-auto px-4 py-8 max-w-7xl">
              {error && (
                <Alert type="error" title="Error" onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {activeView === 'dashboard' && (
                <>
                  <div className="mb-8">
                    <h2 className="text-3xl font-bold text-white mb-2">
                      AI-Powered Strategy Builder
                    </h2>
                    <p className="text-gray-400">
                      Create and test trading strategies using natural language
                    </p>
                  </div>

                  {parsedStrategy && renderParsedStrategy()}
                  {renderStrategyForm()}
                </>
              )}

              {activeView === 'results' && results && (
                <>
                  <div className="mb-8">
                    <h2 className="text-3xl font-bold text-white mb-2">
                      Backtest Results
                    </h2>
                    <p className="text-gray-400">
                      Analysis complete • {results.totalTrades} trades executed
                    </p>
                  </div>

                  {renderResults()}
                </>
              )}

              {activeView === 'history' && (
                <div className="text-center py-16">
                  <Clock className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-400 mb-2">
                    No Previous Backtests
                  </h3>
                  <p className="text-gray-500">
                    Your backtest history will appear here
                  </p>
                </div>
              )}

              {activeView === 'settings' && (
                <div className="max-w-2xl">
                  <h2 className="text-3xl font-bold text-white mb-8">Settings</h2>
                  
                  <div className="space-y-6">
                    <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700">
                      <h3 className="text-lg font-semibold text-white mb-4">Preferences</h3>
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium text-white">Dark Mode</p>
                            <p className="text-sm text-gray-400">Use dark theme throughout the app</p>
                          </div>
                          <button
                            onClick={() => setTheme(isDark ? 'light' : 'dark')}
                            className={`
                              relative inline-flex h-6 w-11 items-center rounded-full
                              ${isDark ? 'bg-blue-600' : 'bg-gray-600'}
                            `}
                          >
                            <span
                              className={`
                                inline-block h-4 w-4 transform rounded-full bg-white transition
                                ${isDark ? 'translate-x-6' : 'translate-x-1'}
                              `}
                            />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </main>
        </div>

        {/* Modals */}
        <Modal
          isOpen={showHelp}
          onClose={() => setShowHelp(false)}
          title="Getting Started"
          size="lg"
        >
          <div className="space-y-4 text-gray-300">
            <div>
              <h4 className="font-semibold text-white mb-2">How to use CryptoBacktest Pro</h4>
              <ol className="list-decimal list-inside space-y-2 text-sm">
                <li>Enter your trading strategy in plain English</li>
                <li>Select the data sources you want to use</li>
                <li>Configure your initial capital and other settings</li>
                <li>Click "Parse Strategy" to convert to trading rules</li>
                <li>Review the parsed rules and click "Run Backtest"</li>
                <li>Analyze your results with professional-grade metrics</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold text-white mb-2">Example Strategies</h4>
              <ul className="space-y-2 text-sm">
                {STRATEGY_EXAMPLES.map((example, i) => (
                  <li key={i} className="flex items-start">
                    <ChevronRight className="w-4 h-4 mr-2 text-gray-500 flex-shrink-0 mt-0.5" />
                    <span>{example}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Modal>

        <Modal
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
          title="Export Results"
        >
          <div className="space-y-4">
            <p className="text-gray-300">
              Choose your preferred export format:
            </p>
            
            <div className="grid grid-cols-2 gap-3">
              {[
                { format: 'PDF', icon: FileText, description: 'Comprehensive report' },
                { format: 'CSV', icon: BarChart3, description: 'Raw trade data' },
                { format: 'JSON', icon: Code, description: 'Complete results' },
                { format: 'Python', icon: Code, description: 'Jupyter notebook' }
              ].map(({ format, icon: Icon, description }) => (
                <button
                  key={format}
                  onClick={() => handleExport(format)}
                  className="p-4 bg-gray-700/50 hover:bg-gray-600/50 rounded-lg 
                           border border-gray-600 transition-all duration-200
                           flex flex-col items-center space-y-2"
                >
                  <Icon className="w-8 h-8 text-gray-400" />
                  <span className="font-medium text-white">{format}</span>
                  <span className="text-xs text-gray-500">{description}</span>
                </button>
              ))}
            </div>
          </div>
        </Modal>

        {/* Progress Overlay */}
        {renderProgress()}
      </div>

      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
        }
        
        .animate-blob {
          animation: blob 7s infinite;
        }
        
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </ErrorBoundary>
  );
};

export default CryptoBacktestDashboard;
