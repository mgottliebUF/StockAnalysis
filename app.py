from flask import Flask, render_template, request, jsonify
import yfinance as yf

app = Flask(__name__)

def get_stock_data(symbol):
    stock_data = yf.Ticker(symbol)
    hist_data = stock_data.history(period="1y", interval="1mo")
    # Calculate the percentage change from the start of the period
    initial_value = hist_data['Close'].iloc[0]
    hist_data['Percentage Change'] = (hist_data['Close'] - initial_value) / initial_value * 100
    return hist_data['Percentage Change'].dropna().tolist(), hist_data.index.strftime('%Y-%m').tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    symbols = request.form.get('symbols', 'SPY')  # Default to SPY if no input
    if 'SPY' not in symbols:
        symbols += ',SPY'  # Always include SPY
    symbol_list = symbols.split(',')
    data = {symbol: get_stock_data(symbol) for symbol in symbol_list}
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
