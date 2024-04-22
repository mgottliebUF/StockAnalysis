import pytest
from app import app, get_stock_data

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Track Year-to-Date Stock Performance" in response.data

def test_get_stock_data():
    """Test the get_stock_data function with a known symbol."""
    result = get_stock_data("AAPL")  # Use a popular stock symbol like Apple
    assert result is not None
    assert result['symbol'] == 'AAPL'
    assert 'start_price' in result
    assert 'latest_price' in result
    assert 'percentage_change' in result

def test_get_stock_data_invalid():
    """Test the get_stock_data function with an invalid symbol."""
    result = get_stock_data("INVALIDSYMBOL")
    assert result is None
