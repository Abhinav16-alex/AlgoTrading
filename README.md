# Algo-Trading System with ML & Automation

## 📋 Assignment Overview
This project implements a complete algorithmic trading system that fetches real stock data, applies trading strategies, uses machine learning predictions, and automatically logs everything to Google Sheets.

## 🎯 Assignment Requirements Completed

### ✅ Data Ingestion (20%)
- **API Used**: Yahoo Finance (yfinance)
- **Stocks Analyzed**: 3 NIFTY 50 stocks (RELIANCE.NS, TCS.NS, INFY.NS)
- **Data Period**: 6 months of daily stock data
- **Data Points**: 124 days per stock with OHLCV data

### ✅ Trading Strategy Logic (20%)
- **Buy Signal**: RSI < 30 (oversold condition)
- **Confirmation**: 20-DMA crossing above 50-DMA
- **Sell Signal**: RSI > 70 OR 20-DMA crossing below 50-DMA
- **Backtest Period**: Complete 6-month analysis
- **Strategy Type**: Conservative, confirmation-based approach

### ✅ Google Sheets Automation (20%)
- **Sheet Name**: StockData
- **Total Tabs**: 5 comprehensive data sheets
  1. **Current_Signals** - Latest buy/sell signals for all stocks
  2. **Backtest_Summary** - Performance metrics and win ratios
  3. **Portfolio_Summary** - Overall portfolio performance
  4. **Detailed_Stock_Data** - Technical indicators for last 30 days
  5. **ML_Accuracy** - Machine learning model performance

### ✅ ML/Analytics (20%)
- **Model Type**: Decision Tree logic with rule-based predictions
- **Features Used**: RSI, MACD, Volume ratios, Price changes
- **Prediction Target**: Next-day price movement (Up/Down)
- **Confidence Scoring**: Based on technical indicator strength
- **Accuracy Tracking**: Comparing predictions vs actual movements

### ✅ Code Quality & Documentation (20%)
- **Structure**: Modular design with clear class separation
- **Logging**: Comprehensive logging throughout execution
- **Error Handling**: Try-catch blocks for robust operation
- **Documentation**: Detailed comments and docstrings
- **Standards**: PEP 8 compliant Python code

## 🚀 How to Run

### Prerequisites
```bash
pip install yfinance pandas numpy scikit-learn gspread google-auth
```

### Setup Requirements
1. **Google Sheets Setup**:
   - Create `StockData` spreadsheet in Google Sheets
   - Set up Google Cloud service account
   - Download credentials as `assignment/creds.json`
   - Share spreadsheet with service account email

2. **File Structure**:
```
project_folder/
├── main.py
├── assignment/
│   └── creds.json
└── trading_output/ (created automatically)
```

### Execution
```bash
python main.py
```

## 📊 Results Summary

### Current Market Analysis (Latest Run)
- **RELIANCE.NS**: SELL signal @ ₹1393.7 (RSI: 25.46)
- **TCS.NS**: HOLD @ ₹3003.0 (RSI: 16.28) 
- **INFY.NS**: HOLD @ ₹1469.6 (RSI: 24.76)

### Strategy Performance
- **Analysis Period**: 6 months (124 trading days)
- **Trading Approach**: Conservative, confirmation-based
- **Risk Management**: Waits for multiple signal confirmations
- **Data Quality**: 100% successful data retrieval for all stocks

### Technical Implementation
- **Real-time Data**: Live fetching from Yahoo Finance API
- **Technical Indicators**: RSI, Moving Averages, MACD, Volume analysis
- **Automation**: Fully automated Google Sheets integration
- **Backup System**: Local CSV files for data redundancy

## 🔧 Technical Architecture

### Core Components
1. **DataFetcher**: Handles API calls and data retrieval
2. **TradingStrategy**: Implements RSI + MA crossover logic
3. **MLPredictor**: Machine learning predictions and accuracy tracking
4. **SheetsManager**: Google Sheets integration and data upload
5. **BacktestEngine**: Historical performance analysis

### Key Features
- **Multi-stock Analysis**: Parallel processing of multiple stocks
- **Real-time Signals**: Live trading signal generation
- **Performance Tracking**: Comprehensive P&L and win ratio analysis
- **Cloud Integration**: Automatic Google Sheets data synchronization
- **Error Recovery**: Robust error handling and logging

## 📈 Business Value

### Risk Management
- Conservative approach prevents excessive trading
- Multiple confirmation signals reduce false positives
- Comprehensive backtesting validates strategy effectiveness

### Automation Benefits
- Eliminates manual data entry errors
- Provides real-time market insights
- Enables scalable analysis across multiple stocks
- Creates audit trail through Google Sheets logging

### Scalability
- Easy to add more stocks to analysis
- Modular design allows strategy modifications
- Cloud-based data storage for team collaboration

## 🎯 Assignment Deliverables

✅ **Complete Python Implementation** (All files included)  
✅ **Google Sheets Integration** (StockData with 5 tabs)  
✅ **6-Month Backtesting** (Historical performance analysis)  
✅ **ML Prediction Model** (Accuracy tracking included)  
✅ **Comprehensive Documentation** (This README + code comments)  
✅ **Real Market Data** (NIFTY 50 stocks analysis)  

## 📞 Contact
- **Developer**: peddireddy shanmukha abhinavi pradeep
- **Assignment**: Algo-Trading System with ML & Automation
- **Completion Date**: August 2, 2025
- **Google Sheet**: StockData (with live updates)

---
*This project demonstrates practical application of algorithmic trading concepts, machine learning integration, and automated data management systems.*
