import time
import warnings
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io
import base64
import yfinance as yf
import json
import os
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning

app = Flask(__name__)

if not os.path.exists('results'):
    os.makedirs('results')

trades = {}
completed_trades = []
total_profit = 0

def create_plot(x, y, predictions, future_days, future_prices, title='', model_type=''):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, predictions, color='red', label='Predicted Line')
    future_x = [x.max() + i for i in future_days]
    plt.scatter(future_x, future_prices, color='green', label='Future Predictions')
    for i, day in enumerate(future_days):
        plt.text(future_x[i], future_prices[i], f'{future_prices[i]:.2f}', color='green')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'{title} ({model_type})')
    plt.legend()
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def fetch_stock_data(ticker, period='1y'):
    df = yf.download(ticker, period=period, interval='1d')
    df = df.reset_index()
    df['days'] = (df['Date'] - df['Date'].min()).dt.days
    return df[['days', 'Close']]

def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'pe_ratio': info.get('trailingPE'),
        'market_cap': info.get('marketCap'),
        'volume': info.get('volume')
    }

def store_results(ticker, model, predictions, future_prices, future_days):
    results = {
        'ticker': ticker,
        'model': model,
        'predictions': predictions.tolist(),
        'future_prices': future_prices,
        'future_days': future_days
    }
    filename = f'results/{ticker}_{model}_{int(time.time())}.json'
    with open(filename, 'w') as f:
        json.dump(results, f)

def calculate_volatility(prices):
    log_returns = np.log(prices / prices.shift(1))
    volatility = np.std(log_returns) * np.sqrt(252)
    return volatility

def calculate_momentum(prices):
    momentum = prices.pct_change().sum()
    return momentum

def calculate_mean_reversion(prices):
    mean_price = prices.mean()
    current_price = prices.iloc[-1]
    mean_reversion = current_price / mean_price
    return mean_reversion

def calculate_volatility_clustering(prices):
    log_returns = np.log(prices / prices.shift(1))
    volatility_clustering = np.mean(log_returns.rolling(window=5).std())
    return volatility_clustering

def calculate_sharpe_ratio(trades):
    returns = [trade['profit'] for trade in trades]
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = avg_return / std_return
    return sharpe_ratio

def fetch_fear_greed_index(ticker):
    return 50

def automatic_analysis(ticker, df):
    latest_price = df['Close'].iloc[-1]
    avg_price = df['Close'].mean()
    analysis = ""
    if latest_price > avg_price:
        analysis = f"The stock {ticker} is trading above its average price over the last year."
    else:
        analysis = f"The stock {ticker} is trading below its average price over the last year."
    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    ticker = request.json.get('ticker', 'AAPL')
    
    df_1y = fetch_stock_data(ticker, '1y')
    df_5y = fetch_stock_data(ticker, '5y')
    
    x_1y = df_1y['days'].values.reshape(-1, 1)
    y_1y = df_1y['Close'].values
    
    x_5y = df_5y['days'].values.reshape(-1, 1)
    y_5y = df_5y['Close'].values

    model_lr_1y = LinearRegression()
    model_lr_1y.fit(x_1y, y_1y)
    predictions_lr_1y = model_lr_1y.predict(x_1y)
    
    model_lr_5y = LinearRegression()
    model_lr_5y.fit(x_5y, y_5y)
    predictions_lr_5y = model_lr_5y.predict(x_5y)
    
    model_svm_1y = SVR(kernel='rbf')
    model_svm_1y.fit(x_1y, y_1y)
    predictions_svm_1y = model_svm_1y.predict(x_1y)
    
    model_svm_5y = SVR(kernel='rbf')
    model_svm_5y.fit(x_5y, y_5y)
    predictions_svm_5y = model_svm_5y.predict(x_5y)
    
    model_knn_1y = KNeighborsRegressor(n_neighbors=5)
    model_knn_1y.fit(x_1y, y_1y)
    predictions_knn_1y = model_knn_1y.predict(x_1y)
    
    model_knn_5y = KNeighborsRegressor(n_neighbors=5)
    model_knn_5y.fit(x_5y, y_5y)
    predictions_knn_5y = model_knn_5y.predict(x_5y)
    
    model_dt_1y = DecisionTreeRegressor()
    model_dt_1y.fit(x_1y, y_1y)
    predictions_dt_1y = model_dt_1y.predict(x_1y)

    model_rf_1y = RandomForestRegressor()
    model_rf_1y.fit(x_1y, y_1y)
    predictions_rf_1y = model_rf_1y.predict(x_1y)

    model_gb_1y = GradientBoostingRegressor()
    model_gb_1y.fit(x_1y, y_1y)
    predictions_gb_1y = model_gb_1y.predict(x_1y)

    model_mlp_1y = MLPRegressor(max_iter=500)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        model_mlp_1y.fit(x_1y, y_1y)
    predictions_mlp_1y = model_mlp_1y.predict(x_1y)

    model_arima_1y = ARIMA(y_1y, order=(5,1,0))
    model_arima_fit_1y = model_arima_1y.fit()
    predictions_arima_1y = model_arima_fit_1y.predict(start=0, end=len(y_1y)-1)

    model_es_1y = ExponentialSmoothing(y_1y, seasonal='mul', seasonal_periods=12)
    model_es_fit_1y = model_es_1y.fit()
    predictions_es_1y = model_es_fit_1y.predict(start=0, end=len(y_1y)-1)
    
    future_days = [30, 60, 90]
    future_prices_lr = [model_lr_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_svm = [model_svm_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_knn = [model_knn_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_dt = [model_dt_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_rf = [model_rf_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_gb = [model_gb_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_mlp = [model_mlp_1y.predict([[x_1y.max() + days]])[0] for days in future_days]
    future_prices_arima = [model_arima_fit_1y.forecast(steps=days)[0] for days in future_days]
    future_prices_es = [model_es_fit_1y.forecast(steps=days)[0] for days in future_days]

    percentage_increase_1m_lr = ((future_prices_lr[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_lr = ((future_prices_lr[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_lr = ((future_prices_lr[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_svm = ((future_prices_svm[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_svm = ((future_prices_svm[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_svm = ((future_prices_svm[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_knn = ((future_prices_knn[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_knn = ((future_prices_knn[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_knn = ((future_prices_knn[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_dt = ((future_prices_dt[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_dt = ((future_prices_dt[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_dt = ((future_prices_dt[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_rf = ((future_prices_rf[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_rf = ((future_prices_rf[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_rf = ((future_prices_rf[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_gb = ((future_prices_gb[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_gb = ((future_prices_gb[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_gb = ((future_prices_gb[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_mlp = ((future_prices_mlp[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_mlp = ((future_prices_mlp[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_mlp = ((future_prices_mlp[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_arima = ((future_prices_arima[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_arima = ((future_prices_arima[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_arima = ((future_prices_arima[2] - y_1y[-1]) / y_1y[-1]) * 100

    percentage_increase_1m_es = ((future_prices_es[0] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_2m_es = ((future_prices_es[1] - y_1y[-1]) / y_1y[-1]) * 100
    percentage_increase_3m_es = ((future_prices_es[2] - y_1y[-1]) / y_1y[-1]) * 100

    plot_url_1y = create_plot(x_1y, y_1y, predictions_lr_1y, future_days, future_prices_lr, f'{ticker} 1 Year Price Prediction', 'Linear Regression')
    plot_url_5y = create_plot(x_5y, y_5y, predictions_lr_5y, [], [], f'{ticker} 5 Years Price Prediction', 'Linear Regression')
    plot_url_svm_1y = create_plot(x_1y, y_1y, predictions_svm_1y, future_days, future_prices_svm, f'{ticker} SVM 1 Year Price Prediction', 'SVM')
    plot_url_svm_5y = create_plot(x_5y, y_5y, predictions_svm_5y, [], [], f'{ticker} SVM 5 Years Price Prediction', 'SVM')
    plot_url_knn_1y = create_plot(x_1y, y_1y, predictions_knn_1y, future_days, future_prices_knn, f'{ticker} KNN 1 Year Price Prediction', 'KNN')
    plot_url_knn_5y = create_plot(x_5y, y_5y, predictions_knn_5y, [], [], f'{ticker} KNN 5 Years Price Prediction', 'KNN')
    plot_url_dt_1y = create_plot(x_1y, y_1y, predictions_dt_1y, future_days, future_prices_dt, f'{ticker} DT 1 Year Price Prediction', 'Decision Tree')
    plot_url_rf_1y = create_plot(x_1y, y_1y, predictions_rf_1y, future_days, future_prices_rf, f'{ticker} RF 1 Year Price Prediction', 'Random Forest')
    plot_url_gb_1y = create_plot(x_1y, y_1y, predictions_gb_1y, future_days, future_prices_gb, f'{ticker} GB 1 Year Price Prediction', 'Gradient Boosting')
    plot_url_mlp_1y = create_plot(x_1y, y_1y, predictions_mlp_1y, future_days, future_prices_mlp, f'{ticker} MLP 1 Year Price Prediction', 'Neural Network')
    plot_url_arima_1y = create_plot(x_1y, y_1y, predictions_arima_1y, future_days, future_prices_arima, f'{ticker} ARIMA 1 Year Price Prediction', 'ARIMA')
    plot_url_es_1y = create_plot(x_1y, y_1y, predictions_es_1y, future_days, future_prices_es, f'{ticker} ES 1 Year Price Prediction', 'Exponential Smoothing')
    
    elapsed_time = time.time() - start_time

    store_results(ticker, 'Linear Regression 1y', predictions_lr_1y, future_prices_lr, future_days)
    store_results(ticker, 'SVM 1y', predictions_svm_1y, future_prices_svm, future_days)
    store_results(ticker, 'SVM 5y', predictions_svm_5y, [], [])
    store_results(ticker, 'KNN 1y', predictions_knn_1y, future_prices_knn, future_days)
    store_results(ticker, 'KNN 5y', predictions_knn_5y, [], [])
    store_results(ticker, 'DT 1y', predictions_dt_1y, future_prices_dt, future_days)
    store_results(ticker, 'RF 1y', predictions_rf_1y, future_prices_rf, future_days)
    store_results(ticker, 'GB 1y', predictions_gb_1y, future_prices_gb, future_days)
    store_results(ticker, 'MLP 1y', predictions_mlp_1y, future_prices_mlp, future_days)
    store_results(ticker, 'ARIMA 1y', predictions_arima_1y, future_prices_arima, future_days)
    store_results(ticker, 'ES 1y', predictions_es_1y, future_prices_es, future_days)

    stock_info = fetch_stock_info(ticker)
    volatility = calculate_volatility(df_1y['Close'])
    momentum = calculate_momentum(df_1y['Close'])
    mean_reversion = calculate_mean_reversion(df_1y['Close'])
    volatility_clustering = calculate_volatility_clustering(df_1y['Close'])
    fear_greed_index = fetch_fear_greed_index(ticker)

    analysis = automatic_analysis(ticker, df_1y)

    response = {
        'plot_url_1y': plot_url_1y,
        'plot_url_5y': plot_url_5y,
        'plot_url_svm_1y': plot_url_svm_1y,
        'plot_url_svm_5y': plot_url_svm_5y,
        'plot_url_knn_1y': plot_url_knn_1y,
        'plot_url_knn_5y': plot_url_knn_5y,
        'plot_url_dt_1y': plot_url_dt_1y,
        'plot_url_rf_1y': plot_url_rf_1y,
        'plot_url_gb_1y': plot_url_gb_1y,
        'plot_url_mlp_1y': plot_url_mlp_1y,
        'plot_url_arima_1y': plot_url_arima_1y,
        'plot_url_es_1y': plot_url_es_1y,
        'intercept_lr': model_lr_1y.intercept_,
        'slope_lr': model_lr_1y.coef_[0],
        'future_price_1m_lr': future_prices_lr[0],
        'future_price_2m_lr': future_prices_lr[1],
        'future_price_3m_lr': future_prices_lr[2],
        'percentage_increase_1m_lr': percentage_increase_1m_lr,
        'percentage_increase_2m_lr': percentage_increase_2m_lr,
        'percentage_increase_3m_lr': percentage_increase_3m_lr,
        'future_price_1m_svm': future_prices_svm[0],
        'future_price_2m_svm': future_prices_svm[1],
        'future_price_3m_svm': future_prices_svm[2],
        'percentage_increase_1m_svm': percentage_increase_1m_svm,
        'percentage_increase_2m_svm': percentage_increase_2m_svm,
        'percentage_increase_3m_svm': percentage_increase_3m_svm,
        'future_price_1m_knn': future_prices_knn[0],
        'future_price_2m_knn': future_prices_knn[1],
        'future_price_3m_knn': future_prices_knn[2],
        'percentage_increase_1m_knn': percentage_increase_1m_knn,
        'percentage_increase_2m_knn': percentage_increase_2m_knn,
        'percentage_increase_3m_knn': percentage_increase_3m_knn,
        'future_price_1m_dt': future_prices_dt[0],
        'future_price_2m_dt': future_prices_dt[1],
        'future_price_3m_dt': future_prices_dt[2],
        'percentage_increase_1m_dt': percentage_increase_1m_dt,
        'percentage_increase_2m_dt': percentage_increase_2m_dt,
        'percentage_increase_3m_dt': percentage_increase_3m_dt,
        'future_price_1m_rf': future_prices_rf[0],
        'future_price_2m_rf': future_prices_rf[1],
        'future_price_3m_rf': future_prices_rf[2],
        'percentage_increase_1m_rf': percentage_increase_1m_rf,
        'percentage_increase_2m_rf': percentage_increase_2m_rf,
        'percentage_increase_3m_rf': percentage_increase_3m_rf,
        'future_price_1m_gb': future_prices_gb[0],
        'future_price_2m_gb': future_prices_gb[1],
        'future_price_3m_gb': future_prices_gb[2],
        'percentage_increase_1m_gb': percentage_increase_1m_gb,
        'percentage_increase_2m_gb': percentage_increase_2m_gb,
        'percentage_increase_3m_gb': percentage_increase_3m_gb,
        'future_price_1m_mlp': future_prices_mlp[0],
        'future_price_2m_mlp': future_prices_mlp[1],
        'future_price_3m_mlp': future_prices_mlp[2],
        'percentage_increase_1m_mlp': percentage_increase_1m_mlp,
        'percentage_increase_2m_mlp': percentage_increase_2m_mlp,
        'percentage_increase_3m_mlp': percentage_increase_3m_mlp,
        'future_price_1m_arima': future_prices_arima[0],
        'future_price_2m_arima': future_prices_arima[1],
        'future_price_3m_arima': future_prices_arima[2],
        'percentage_increase_1m_arima': percentage_increase_1m_arima,
        'percentage_increase_2m_arima': percentage_increase_2m_arima,
        'percentage_increase_3m_arima': percentage_increase_3m_arima,
        'future_price_1m_es': future_prices_es[0],
        'future_price_2m_es': future_prices_es[1],
        'future_price_3m_es': future_prices_es[2],
        'percentage_increase_1m_es': percentage_increase_1m_es,
        'percentage_increase_2m_es': percentage_increase_2m_es,
        'percentage_increase_3m_es': percentage_increase_3m_es,
        'volatility': volatility,
        'momentum': momentum,
        'mean_reversion': mean_reversion,
        'volatility_clustering': volatility_clustering,
        'fear_greed_index': fear_greed_index,
        'pe_ratio': stock_info['pe_ratio'],
        'market_cap': stock_info['market_cap'],
        'volume': stock_info['volume'],
        'analysis': analysis,
        'elapsed_time': elapsed_time
    }
    return jsonify(response)

@app.route('/trades', methods=['POST'])
def add_trade():
    data = request.json
    ticker = data['ticker']
    buy_price = data['buy_price']
    shares = data['shares']
    trades[ticker] = {'buy_price': buy_price, 'shares': shares, 'current_price': None, 'percentage_change': None, 'profit': None}
    return jsonify({'status': 'success', 'message': 'Trade added successfully'})

@app.route('/trades', methods=['GET'])
def get_trades():
    global total_profit
    total_profit = 0
    trade_info = []
    for ticker, trade in trades.items():
        current_price = yf.download(ticker, period='1d', interval='1m')['Close'].iloc[-1]
        percentage_change = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
        profit = (current_price - trade['buy_price']) * trade['shares']
        total_profit += profit
        trades[ticker]['current_price'] = current_price
        trades[ticker]['percentage_change'] = percentage_change
        trades[ticker]['profit'] = profit
        trade_info.append({
            'ticker': ticker,
            'buy_price': trade['buy_price'],
            'current_price': current_price,
            'percentage_change': percentage_change,
            'profit': profit,
            'shares': trade['shares']
        })
    return jsonify({'trade_info': trade_info, 'total_profit': total_profit})

@app.route('/trades/clear', methods=['POST'])
def clear_trades():
    trades.clear()
    completed_trades.clear()
    global total_profit
    total_profit = 0
    return jsonify({'status': 'success', 'message': 'Trade log cleared successfully'})

@app.route('/trades/complete', methods=['POST'])
def complete_trade():
    data = request.json
    ticker = data['ticker']
    if ticker in trades:
        completed_trades.append(trades.pop(ticker))
        return jsonify({'status': 'success', 'message': f'Trade for {ticker} completed successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Trade not found'})

@app.route('/trades/sharpe', methods=['GET'])
def sharpe_ratio():
    if completed_trades:
        sharpe_ratio = calculate_sharpe_ratio(completed_trades)
        return jsonify({'sharpe_ratio': sharpe_ratio})
    else:
        return jsonify({'status': 'error', 'message': 'No completed trades found'})

@app.route('/trades/total_profit', methods=['POST'])
def reset_total_profit():
    global total_profit
    total_profit = 0
    return jsonify({'status': 'success', 'message': 'Total profit reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)

