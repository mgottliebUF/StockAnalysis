from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime, timedelta, date
import json
import pika
import requests
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class StockSearch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbols = db.Column(db.String(100), nullable=False)
    search_results = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<StockSearch {self.symbols}>'

with app.app_context():
    db.create_all()

def predict_future_prices(hist_data, months=6):
    dates = np.arange(len(hist_data)).reshape(-1, 1)
    prices = hist_data['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(dates, prices)
    future_dates = np.arange(len(hist_data), len(hist_data) + months).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    return predicted_prices.flatten()

def get_stock_data(symbol):
    stock_data = yf.Ticker(symbol)
    hist_data = stock_data.history(period="1y", interval="1mo")
    future_prices = predict_future_prices(hist_data)
    initial_value = hist_data['Close'].iloc[0]
    hist_percentage_change = ((hist_data['Close'] - initial_value) / initial_value * 100).dropna()
    hist_dates = hist_data.index.strftime('%Y-%m').tolist()

    if hist_dates[-1] != date.today().strftime('%Y-%m'):
        hist_dates.append(date.today().strftime('%Y-%m'))
        hist_percentage_change = np.append(hist_percentage_change, hist_percentage_change[-1])

    future_dates = [(date.today() + timedelta(days=30*i)).strftime('%Y-%m') for i in range(1, 7)]

    return hist_percentage_change.tolist(), hist_dates, future_prices.tolist(), future_dates

def connect_rabbitmq():
    credentials = pika.PlainCredentials('guest', 'guest')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', credentials=credentials))
    channel = connection.channel()
    channel.queue_declare(queue='stock_searches')
    return channel

def send_message_to_queue(symbol_data):
    channel = connect_rabbitmq()
    channel.basic_publish(exchange='',
                          routing_key='stock_searches',
                          body=json.dumps(symbol_data))
    channel.close()

def consume_messages():
    channel = connect_rabbitmq()

    def callback(ch, method, properties, body):
        print("Received %r" % json.loads(body))

    channel.basic_consume(queue='stock_searches', on_message_callback=callback, auto_ack=True)
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

@app.route('/', methods=['GET', 'POST'])
def home():
    data = {}
    real_time_data = False
    news_articles = []
    if request.method == 'POST':
        symbols = request.form.get('symbols', 'SPY')
        symbol_list = symbols.split(',')
        data = {symbol: get_stock_data(symbol) for symbol in symbol_list}
        search_results_json = json.dumps(data)
        new_search = StockSearch(symbols=symbols, search_results=search_results_json)
        db.session.add(new_search)
        db.session.commit()
        send_message_to_queue(data)
        real_time_data = True
        news_articles = fetch_news(",".join(symbol_list))

    last_three_searches = StockSearch.query.order_by(StockSearch.id.desc()).limit(3).all()
    last_searches_data = [
        (search.symbols, json.loads(search.search_results), search.created_at.strftime('%Y-%m-%d %H:%M:%S')) for search in last_three_searches
    ]

    return render_template('index.html', data=data, last_searches=last_searches_data, real_time_data=real_time_data, news_articles=news_articles)

def fetch_news(query='stock market'):
    gnews_api_key = "4dcc849f049fed0e90dc4ac6da14830f"  # Replace with your actual GNEWS API key
    gnews_endpoint = "https://gnews.io/api/v4/search"
    response = requests.get(gnews_endpoint, params={'q': query, 'token': gnews_api_key, 'lang': 'en'})
    news_data = response.json()
    return news_data.get('articles', [])[:3]

if __name__ == '__main__':
    app.run(debug=True)
