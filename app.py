from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class StockSearch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbols = db.Column(db.String(100), nullable=False)
    search_results = db.Column(db.PickleType, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<StockSearch {self.symbols}>'

# Ensure all models are imported or defined before this line
with app.app_context():
    db.drop_all()  # This will drop all tables
    db.create_all()  # This will create tables based on the current model definitions

def get_stock_data(symbol):
    stock_data = yf.Ticker(symbol)
    hist_data = stock_data.history(period="1y", interval="1mo")
    initial_value = hist_data['Close'].iloc[0]
    hist_data['Percentage Change'] = (hist_data['Close'] - initial_value) / initial_value * 100
    return hist_data['Percentage Change'].dropna().tolist(), hist_data.index.strftime('%Y-%m').tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    real_time_data = False
    if request.method == 'POST':
        symbols = request.form.get('symbols', 'SPY')
        if 'SPY' not in symbols:
            symbols += ',SPY'
        symbol_list = symbols.split(',')
        data = {symbol: get_stock_data(symbol) for symbol in symbol_list}

        # Save search results to the database
        new_search = StockSearch(symbols=symbols, search_results=data)
        db.session.add(new_search)
        db.session.commit()
        real_time_data = True

    # Retrieve the last three searches from the database
    last_three_searches = StockSearch.query.order_by(StockSearch.id.desc()).limit(3).all()

    return render_template('index.html', data=data if request.method == 'POST' else None, 
                           last_searches=last_three_searches, real_time_data=real_time_data)

if __name__ == '__main__':
    app.run(debug=True)
