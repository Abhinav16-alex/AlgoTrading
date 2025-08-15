# main.py
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import logging
import time
import os
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Sheets Manager Class
class GoogleSheetsManager:
    def __init__(self, credentials_file='assignment/creds.json', sheet_name='StockData'):
        """
        Initialize Google Sheets connection using your existing setup
        """
        try:
            self.scope = ['https://spreadsheets.google.com/feeds',
                         'https://www.googleapis.com/auth/drive']
            
            self.creds = Credentials.from_service_account_file(credentials_file, scopes=self.scope)
            self.client = gspread.authorize(self.creds)
            self.sheet_name = sheet_name
            
            # Open existing spreadsheet
            try:
                self.spreadsheet = self.client.open(sheet_name)
                logging.info(f"✅ Connected to existing Google Sheet: {sheet_name}")
            except gspread.SpreadsheetNotFound:
                logging.error(f"❌ Spreadsheet '{sheet_name}' not found. Please create it first.")
                self.spreadsheet = None
                
        except Exception as e:
            logging.error(f"❌ Failed to connect to Google Sheets: {str(e)}")
            self.spreadsheet = None
    
    def update_sheet(self, worksheet_name, data_df):
        """Update specific worksheet with DataFrame"""
        if not self.spreadsheet:
            logging.error("❌ No spreadsheet connection")
            return False
            
        try:
            # Create or get worksheet
            try:
                worksheet = self.spreadsheet.worksheet(worksheet_name)
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            
            # Convert DataFrame to list of lists for Google Sheets
            data_to_update = [data_df.columns.values.tolist()] + data_df.values.tolist()
            
            # Update with data
            worksheet.update(data_to_update)
            logging.info(f"✅ Updated Google Sheet tab: {worksheet_name}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error updating sheet {worksheet_name}: {str(e)}")
            return False
    
    def update_all_sheets(self, data_dict):
        """Update multiple worksheets at once"""
        success_count = 0
        
        for sheet_name, data_df in data_dict.items():
            if self.update_sheet(sheet_name, data_df):
                success_count += 1
        
        logging.info(f"✅ Successfully updated {success_count}/{len(data_dict)} sheets")
        return success_count == len(data_dict)

class AlgoTradingSystem:
    def __init__(self):
        # NIFTY 50 stocks (top 3 for demo)
        self.stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        self.portfolio = {stock: {'position': 0, 'avg_price': 0, 'total_invested': 0, 'current_value': 0} for stock in self.stocks}
        self.trade_log = []
        self.all_signals = []
        self.backtest_results = {}
        
        # Initialize Google Sheets manager
        self.sheets_manager = GoogleSheetsManager()
        
    def run_system(self):
        """Main system execution"""
        logging.info("🚀 Starting Complete Algo-Trading System...")
        
        try:
            # 1. Data Ingestion for all stocks
            stock_data = self.fetch_all_data()
            
            # 2. Generate signals and ML predictions
            self.analyze_all_stocks(stock_data)
            
            # 3. Run backtest
            self.run_backtest(stock_data)
            
            # 4. Generate comprehensive reports
            self.generate_reports()
            
            # 5. Save all data to CSV (simulating Google Sheets)
            self.save_to_sheets()
            
            logging.info("✅ System execution completed successfully!")
            
        except Exception as e:
            logging.error(f"❌ System error: {str(e)}")
    
    def fetch_all_data(self):
        """Fetch 6 months data for all stocks"""
        stock_data = {}
        
        for stock in self.stocks:
            try:
                logging.info(f"📥 Fetching data for {stock}...")
                ticker = yf.Ticker(stock)
                data = ticker.history(period="6mo", interval="1d")
                
                if len(data) > 0:
                    stock_data[stock] = data
                    logging.info(f"✅ Fetched {len(data)} days of data for {stock}")
                else:
                    logging.warning(f"⚠️ No data available for {stock}")
                    
            except Exception as e:
                logging.error(f"❌ Failed to fetch data for {stock}: {str(e)}")
        
        return stock_data
    
    def calculate_technical_indicators(self, data):
        """Calculate RSI, Moving Averages, MACD"""
        df = data.copy()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def generate_trading_signals(self, data, stock):
        """Generate buy/sell signals based on strategy"""
        df = self.calculate_technical_indicators(data)
        
        # Trading Strategy Implementation
        # Buy Signal: RSI < 30 AND 20-MA crosses above 50-MA
        df['RSI_Buy'] = df['RSI'] < 30
        df['MA_Crossover'] = (df['MA_20'] > df['MA_50']) & (df['MA_20'].shift(1) <= df['MA_50'].shift(1))
        
        # Sell Signal: RSI > 70 OR 20-MA crosses below 50-MA
        df['RSI_Sell'] = df['RSI'] > 70
        df['MA_Crossunder'] = (df['MA_20'] < df['MA_50']) & (df['MA_20'].shift(1) >= df['MA_50'].shift(1))
        
        # Final Signals
        df['Buy_Signal'] = df['RSI_Buy'] & df['MA_Crossover']
        df['Sell_Signal'] = df['RSI_Sell'] | df['MA_Crossunder']
        
        # Create signal summary
        df['Signal'] = 'HOLD'
        df.loc[df['Buy_Signal'], 'Signal'] = 'BUY'
        df.loc[df['Sell_Signal'], 'Signal'] = 'SELL'
        
        return df
    
    def ml_prediction(self, data):
        """Simple ML prediction using decision tree logic"""
        df = self.calculate_technical_indicators(data)
        
        # Create features for prediction
        df['Price_Change'] = df['Close'].pct_change()
        df['Next_Day_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Simple rule-based prediction (simulating ML)
        conditions = [
            (df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal']),  # Oversold + MACD bullish
            (df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal']),  # Overbought + MACD bearish
        ]
        choices = [1, 0]  # 1 = Up, 0 = Down
        
        df['ML_Prediction'] = np.select(conditions, choices, default=0.5)
        df['ML_Confidence'] = np.where(
            (df['RSI'] < 30) | (df['RSI'] > 70), 0.8, 0.6
        )
        
        return df
    
    def analyze_all_stocks(self, stock_data):
        """Analyze all stocks and generate signals"""
        for stock, data in stock_data.items():
            try:
                logging.info(f"📊 Analyzing {stock}...")
                
                # Generate signals
                signals_df = self.generate_trading_signals(data, stock)
                
                # Add ML predictions
                ml_df = self.ml_prediction(data)
                signals_df['ML_Prediction'] = ml_df['ML_Prediction']
                signals_df['ML_Confidence'] = ml_df['ML_Confidence']
                
                # Store latest signals
                latest_signal = {
                    'Stock': stock,
                    'Date': signals_df.index[-1].strftime('%Y-%m-%d'),
                    'Current_Price': round(signals_df['Close'].iloc[-1], 2),
                    'RSI': round(signals_df['RSI'].iloc[-1], 2),
                    'MA_20': round(signals_df['MA_20'].iloc[-1], 2),
                    'MA_50': round(signals_df['MA_50'].iloc[-1], 2),
                    'MACD': round(signals_df['MACD'].iloc[-1], 4),
                    'Volume_Ratio': round(signals_df['Volume_Ratio'].iloc[-1], 2),
                    'Signal': signals_df['Signal'].iloc[-1],
                    'ML_Prediction': signals_df['ML_Prediction'].iloc[-1],
                    'ML_Confidence': round(signals_df['ML_Confidence'].iloc[-1], 2)
                }
                
                self.all_signals.append(latest_signal)
                
                # Store full data for backtesting
                self.backtest_results[stock] = signals_df
                
                logging.info(f"✅ Analysis completed for {stock} - Signal: {latest_signal['Signal']}")
                
            except Exception as e:
                logging.error(f"❌ Error analyzing {stock}: {str(e)}")
    
    def run_backtest(self, stock_data):
        """Run 6-month backtest for all stocks"""
        logging.info("🔄 Running 6-month backtest...")
        
        backtest_summary = []
        
        for stock, signals_df in self.backtest_results.items():
            try:
                initial_capital = 100000  # 1 Lakh per stock
                position = 0
                capital = initial_capital
                trades = []
                
                for date, row in signals_df.iterrows():
                    if row['Signal'] == 'BUY' and position == 0:
                        # Buy
                        shares = capital // row['Close']
                        if shares > 0:
                            position = shares
                            buy_price = row['Close']
                            capital -= shares * buy_price
                            trades.append({
                                'Date': date,
                                'Action': 'BUY',
                                'Price': buy_price,
                                'Shares': shares,
                                'Value': shares * buy_price
                            })
                    
                    elif row['Signal'] == 'SELL' and position > 0:
                        # Sell
                        sell_value = position * row['Close']
                        capital += sell_value
                        profit = sell_value - (position * buy_price)
                        trades.append({
                            'Date': date,
                            'Action': 'SELL',
                            'Price': row['Close'],
                            'Shares': position,
                            'Value': sell_value,
                            'Profit': profit
                        })
                        position = 0
                
                # Calculate final portfolio value
                final_value = capital
                if position > 0:
                    final_value += position * signals_df['Close'].iloc[-1]
                
                total_return = ((final_value - initial_capital) / initial_capital) * 100
                winning_trades = len([t for t in trades if t.get('Profit', 0) > 0])
                total_trades = len([t for t in trades if t['Action'] == 'SELL'])
                win_ratio = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                backtest_summary.append({
                    'Stock': stock,
                    'Initial_Capital': initial_capital,
                    'Final_Value': round(final_value, 2),
                    'Total_Return_%': round(total_return, 2),
                    'Total_Trades': total_trades,
                    'Winning_Trades': winning_trades,
                    'Win_Ratio_%': round(win_ratio, 2),
                    'Trade_Count': len(trades)
                })
                
                # Store trades for sheets
                for trade in trades:
                    trade['Stock'] = stock
                    self.trade_log.append(trade)
                
                logging.info(f"✅ Backtest completed for {stock} - Return: {total_return:.2f}%")
                
            except Exception as e:
                logging.error(f"❌ Backtest error for {stock}: {str(e)}")
        
        self.backtest_summary = backtest_summary
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        logging.info("📋 Generating comprehensive reports...")
        
        # Portfolio Summary
        total_return = sum([s['Total_Return_%'] for s in self.backtest_summary])
        avg_return = total_return / len(self.backtest_summary) if self.backtest_summary else 0
        total_trades = sum([s['Total_Trades'] for s in self.backtest_summary])
        avg_win_ratio = sum([s['Win_Ratio_%'] for s in self.backtest_summary]) / len(self.backtest_summary) if self.backtest_summary else 0
        
        self.portfolio_summary = {
            'Total_Stocks_Analyzed': len(self.stocks),
            'Average_Return_%': round(avg_return, 2),
            'Total_Trades_Executed': total_trades,
            'Average_Win_Ratio_%': round(avg_win_ratio, 2),
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Backtest_Period': '6 months'
        }
        
        logging.info(f"✅ Portfolio Summary: Avg Return: {avg_return:.2f}%, Win Ratio: {avg_win_ratio:.2f}%")
    
    def save_to_sheets(self):
        """Save all data to Google Sheets and CSV files"""
        logging.info("💾 Saving data to Google Sheets and local files...")
        
        try:
            # Create output directory for local backup
            os.makedirs('trading_output', exist_ok=True)
            
            # Prepare all data for sheets
            sheets_data = {}
            
            # Tab 1: Current Signals
            signals_df = pd.DataFrame(self.all_signals)
            signals_df.to_csv('trading_output/01_current_signals.csv', index=False)
            sheets_data['Current_Signals'] = signals_df
            
            # Tab 2: Trade Log
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv('trading_output/02_trade_log.csv', index=False)
                sheets_data['Trade_Log'] = trades_df
            
            # Tab 3: Backtest Summary
            backtest_df = pd.DataFrame(self.backtest_summary)
            backtest_df.to_csv('trading_output/03_backtest_summary.csv', index=False)
            sheets_data['Backtest_Summary'] = backtest_df
            
            # Tab 4: Portfolio Summary
            portfolio_df = pd.DataFrame([self.portfolio_summary])
            portfolio_df.to_csv('trading_output/04_portfolio_summary.csv', index=False)
            sheets_data['Portfolio_Summary'] = portfolio_df
            
            # Tab 5: Detailed Stock Data (last 30 days for each stock)
            detailed_data = []
            for stock, df in self.backtest_results.items():
                recent_data = df.tail(30).copy()
                recent_data['Stock'] = stock
                recent_data['Date'] = recent_data.index.strftime('%Y-%m-%d')
                cols_to_include = ['Stock', 'Date', 'Close', 'RSI', 'MA_20', 'MA_50', 'MACD', 'Volume', 'Signal']
                detailed_data.append(recent_data[cols_to_include])
            
            if detailed_data:
                all_detailed = pd.concat(detailed_data, ignore_index=True)
                all_detailed.to_csv('trading_output/05_detailed_stock_data.csv', index=False)
                sheets_data['Detailed_Stock_Data'] = all_detailed
            
            # Tab 6: ML Predictions Accuracy
            ml_accuracy = []
            for stock, df in self.backtest_results.items():
                df_clean = df.dropna()
                if len(df_clean) > 1:
                    actual_moves = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
                    predicted_moves = (df_clean['ML_Prediction'] > 0.5).astype(int)
                    accuracy = (actual_moves == predicted_moves).mean() * 100
                    
                    ml_accuracy.append({
                        'Stock': stock,
                        'ML_Accuracy_%': round(accuracy, 2),
                        'Total_Predictions': len(df_clean),
                        'Avg_Confidence': round(df_clean['ML_Confidence'].mean(), 2)
                    })
            
            if ml_accuracy:
                ml_df = pd.DataFrame(ml_accuracy)
                ml_df.to_csv('trading_output/06_ml_accuracy.csv', index=False)
                sheets_data['ML_Accuracy'] = ml_df
            
            # Update Google Sheets
            if self.sheets_manager.spreadsheet:
                logging.info("📊 Updating Google Sheets...")
                success = self.sheets_manager.update_all_sheets(sheets_data)
                if success:
                    logging.info("✅ All data successfully uploaded to Google Sheets: StockData")
                else:
                    logging.warning("⚠️ Some sheets failed to update. Check local CSV files.")
            else:
                logging.warning("⚠️ Google Sheets not connected. Data saved to local CSV files only.")
            
            logging.info("✅ All data saved successfully!")
            logging.info("📁 Local backup files created in trading_output/")
            logging.info("📊 Google Sheets updated: StockData")
            
        except Exception as e:
            logging.error(f"❌ Error saving data: {str(e)}")
            logging.info("📁 Local CSV files may still be available in trading_output/")
    
    def display_summary(self):
        """Display summary in console"""
        print("\n" + "="*60)
        print("🎯 ALGO-TRADING SYSTEM SUMMARY")
        print("="*60)
        
        print(f"📅 Analysis Date: {self.portfolio_summary['Analysis_Date']}")
        print(f"📊 Stocks Analyzed: {self.portfolio_summary['Total_Stocks_Analyzed']}")
        print(f"📈 Average Return: {self.portfolio_summary['Average_Return_%']}%")
        print(f"🎯 Win Ratio: {self.portfolio_summary['Average_Win_Ratio_%']}%")
        print(f"💼 Total Trades: {self.portfolio_summary['Total_Trades_Executed']}")
        
        print("\n📋 CURRENT SIGNALS:")
        for signal in self.all_signals:
            status_emoji = "🟢" if signal['Signal'] == 'BUY' else "🔴" if signal['Signal'] == 'SELL' else "🟡"
            print(f"{status_emoji} {signal['Stock']}: {signal['Signal']} @ ₹{signal['Current_Price']} (RSI: {signal['RSI']})")
        
        print("\n💰 BACKTEST RESULTS:")
        for result in self.backtest_summary:
            print(f"📈 {result['Stock']}: {result['Total_Return_%']}% return, {result['Win_Ratio_%']}% win rate")
        
        print("\n✅ All detailed data saved to 'trading_output' folder")
        print("="*60)

def main():
    """Main execution function"""
    system = AlgoTradingSystem()
    system.run_system()
    system.display_summary()

if __name__ == "__main__":
    main()


