import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app import app, db, StockSearch, get_stock_data, predict_future_prices

class TestPredictFuturePrices(unittest.TestCase):
    @patch('app.LinearRegression')
    def test_predict_future_prices(self, mock_lr):
        mock_model = MagicMock()
        mock_lr.return_value = mock_model
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([130, 135, 140])

        hist_data = pd.DataFrame({
            'Close': [100, 105, 110, 115, 120]
        })
        result = predict_future_prices(hist_data)
        self.assertEqual(list(result), [130, 135, 140])

class TestGetStockData(unittest.TestCase):
    @patch('app.yf.Ticker')
    def test_get_stock_data(self, mock_ticker):
        mock_history = pd.DataFrame({
            'Close': [100, 105, 110, 115, 120]
        }, index=pd.date_range(start='1/1/2020', periods=5, freq='M'))
        mock_ticker.return_value.history.return_value = mock_history

        hist_percentage_change, hist_dates, future_prices, future_dates = get_stock_data('AAPL')
        self.assertIsInstance(hist_percentage_change, list)
        self.assertIsInstance(hist_dates, list)
        self.assertIsInstance(future_prices, list)
        self.assertIsInstance(future_dates, list)

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_home_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    @patch('app.get_stock_data')
    @patch('app.StockSearch')
    @patch('app.db.session')
    def test_home_post(self, mock_db_session, mock_StockSearch, mock_get_stock_data):
        mock_get_stock_data.return_value = ([10, 15, 20], ['2020-01', '2020-02', '2020-03'], [25, 30], ['2020-04', '2020-05'])
        mock_StockSearch.return_value = MagicMock()

        response = self.client.post('/', data={'symbols': 'AAPL,GOOGL'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
