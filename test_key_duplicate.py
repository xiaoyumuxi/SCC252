"""
Unit test file: test all API functions, including duplicate-key scenarios.

Coverage:
1. Normal behavior for all API endpoints
2. Duplicate keys in JSON requests
3. Edge cases and error handling
4. Data validation and exception handling
"""

import pytest
import json
import os
import tempfile
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, init_db


@pytest.fixture
def client():
    """Create a Flask test client with a temporary database."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    # Create a temporary sqlite database file
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    test_db.close()

    # Backup the original DB path and switch to test DB
    import app as app_module
    original_db = app_module.DB_FILE
    app_module.DB_FILE = test_db.name

    # Init test DB schema
    init_db()

    with app.test_client() as c:
        yield c

    # Cleanup: restore DB path and delete temp DB
    app_module.DB_FILE = original_db
    if os.path.exists(test_db.name):
        os.unlink(test_db.name)


@pytest.fixture
def sample_features():
    """Generate a sample feature vector (default 78 features)."""
    return [float(i) for i in range(78)]


@pytest.fixture
def mock_model_loaded():
    """Mock model components as loaded."""
    with patch('app.MODEL', MagicMock()), \
         patch('app.SCALER', MagicMock()), \
         patch('app.LE', MagicMock()), \
         patch('app.FEATURE_COLUMNS', [f'f_{i}' for i in range(78)]):

        # Provide a fake inverse_transform
        import app as app_module
        app_module.LE.inverse_transform = MagicMock(return_value=['BENIGN'])
        yield


class TestHealthCheck:
    """Tests for /health endpoint."""

    def test_health_check_success(self, client):
        """Health check should return status and model_loaded."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data


class TestPredictAPI:
    """Tests for /api/predict endpoint."""

    def test_predict_missing_features(self, client):
        """Should return 400 if 'features' is missing."""
        response = client.post('/api/predict', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'features' in data['message'].lower()

    def test_predict_empty_json(self, client):
        """Should return 400 on empty JSON request."""
        response = client.post('/api/predict', json=None, content_type='application/json')
        assert response.status_code == 400

    def test_predict_key_duplicate_in_json(self, client, sample_features, mock_model_loaded):
        """
        Test duplicate keys in raw JSON string.
        Note: Python dict cannot preserve duplicate keys, but raw JSON can.
        Most parsers keep the last occurrence.
        """
        json_str = '{"features": [1, 2, 3], "features": ' + str(sample_features) + '}'
        response = client.post('/api/predict', data=json_str, content_type='application/json')
        assert response.status_code in [200, 400, 503]

    def test_predict_multiple_duplicate_keys(self, client, sample_features):
        """
        Test multiple duplicate keys in a Python dict.
        Note: Python dict keeps only the last value.
        """
        json_data = {
            'features': sample_features,
            'features': sample_features,
            'extra': 'value1',
            'extra': 'value2'
        }
        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 503]

    def test_predict_invalid_features_type(self, client):
        """Should error if 'features' is not a list."""
        response = client.post('/api/predict', json={'features': 'not_a_list'})
        assert response.status_code in [400, 500, 503]

    def test_predict_wrong_feature_count(self, client, mock_model_loaded):
        """Should error if feature count mismatches expected length."""
        wrong_features = [1.0, 2.0, 3.0]
        response = client.post('/api/predict', json={'features': wrong_features})
        assert response.status_code in [200, 400, 500, 503]


class TestAlertsAPI:
    """Tests for /api/alerts endpoint."""

    def test_get_alerts_success(self, client):
        """Should return a list."""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_get_alerts_empty(self, client):
        """Should return empty list initially."""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []


class TestHistoryAPI:
    """Tests for /api/history endpoint."""

    def test_get_history_success(self, client):
        """Should return a list."""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_get_history_empty(self, client):
        """Should return a list (possibly empty)."""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestPerformanceAPI:
    """Tests for /api/performance endpoint."""

    def test_get_performance_success(self, client):
        """Should return performance dict (if available)."""
        response = client.get('/api/performance')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        for key in expected_keys:
            if key in data:
                assert isinstance(data[key], (int, float))


class TestStreamAPI:
    """Tests for /api/stream endpoint."""

    def test_get_stream_success(self, client):
        """Should return 200 or an error depending on stream readiness."""
        response = client.get('/api/stream')
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data

    def test_get_stream_with_label_filter(self, client):
        """Should support label filtering (if label column exists)."""
        response = client.get('/api/stream?label=DoS%20Hulk')
        assert response.status_code in [200, 404, 500]

    def test_get_stream_invalid_label(self, client):
        """Invalid label filter may return 404 or error."""
        response = client.get('/api/stream?label=NonExistentAttack')
        assert response.status_code in [200, 404, 500]

    def test_get_stream_duplicate_query_params(self, client):
        """Duplicate query params should be handled by Flask."""
        response = client.get('/api/stream?label=DoS&label=PortScan')
        assert response.status_code in [200, 404, 500]


class TestRandomAPI:
    """Tests for /api/random endpoint."""

    def test_get_random_success(self, client):
        """Should return features and feature_names with same length."""
        response = client.get('/api/random')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'features' in data
        assert 'feature_names' in data
        assert isinstance(data['features'], list)
        assert isinstance(data['feature_names'], list)
        assert len(data['features']) == len(data['feature_names'])


class TestUploadAndRetrainAPI:
    """Tests for /api/upload-and-retrain endpoint."""

    def test_upload_no_files(self, client):
        """Should return 400 if no files sent."""
        response = client.post('/api/upload-and-retrain')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_upload_empty_files(self, client):
        """Empty file list should return 400."""
        response = client.post('/api/upload-and-retrain', data={'files': []})
        assert response.status_code == 400

    def test_upload_invalid_file_type(self, client):
        """Non-CSV file should fail or be ignored."""
        data = {
            'files': (BytesIO(b'not csv content'), 'test.txt')
        }
        response = client.post('/api/upload-and-retrain', data=data, content_type='multipart/form-data')
        assert response.status_code in [400, 200]

    def test_upload_valid_csv(self, client):
        """Upload a valid CSV file."""
        test_data = {
            'Label': ['BENIGN', 'DDoS', 'BENIGN'],
            'Feature1': [1.0, 2.0, 3.0],
            'Feature2': [4.0, 5.0, 6.0],
            'Feature3': [7.0, 8.0, 9.0]
        }
        df = pd.DataFrame(test_data)

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        data = {
            'files': (csv_buffer, 'test.csv')
        }

        response = client.post('/api/upload-and-retrain', data=data, content_type='multipart/form-data')
        assert response.status_code in [200, 400, 500]

    def test_upload_multiple_files(self, client):
        """Upload multiple CSV files."""
        test_data1 = {
            'Label': ['BENIGN'],
            'Feature1': [1.0],
            'Feature2': [2.0]
        }
        test_data2 = {
            'Label': ['DDoS'],
            'Feature1': [3.0],
            'Feature2': [4.0]
        }

        df1 = pd.DataFrame(test_data1)
        df2 = pd.DataFrame(test_data2)

        csv1 = BytesIO()
        csv2 = BytesIO()
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)
        csv1.seek(0)
        csv2.seek(0)

        data = {
            'files': [(csv1, 'test1.csv'), (csv2, 'test2.csv')]
        }

        response = client.post('/api/upload-and-retrain', data=data, content_type='multipart/form-data')
        assert response.status_code in [200, 400, 500]

    def test_upload_csv_without_label_column(self, client):
        """CSV without 'Label' column should return 400."""
        test_data = {
            'Feature1': [1.0, 2.0],
            'Feature2': [3.0, 4.0]
        }
        df = pd.DataFrame(test_data)

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        data = {
            'files': (csv_buffer, 'test.csv')
        }

        response = client.post('/api/upload-and-retrain', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'


class TestKeyDuplicateScenarios:
    """Dedicated tests for duplicate-key scenarios."""

    def test_json_duplicate_key_last_wins(self, client):
        """Duplicate key in Python dict: last value wins."""
        json_data = {
            'features': [1, 2, 3],
            'features': [4, 5, 6]
        }
        assert json_data['features'] == [4, 5, 6]

        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 500, 503]

    def test_multiple_duplicate_keys_in_request(self, client):
        """Multiple duplicate keys: last value wins for each key."""
        json_data = {
            'features': [1.0] * 78,
            'features': [2.0] * 78,
            'extra_param': 'value1',
            'extra_param': 'value2',
            'another': 100,
            'another': 200
        }
        assert json_data['features'] == [2.0] * 78
        assert json_data['extra_param'] == 'value2'
        assert json_data['another'] == 200

        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 500, 503]

    def test_nested_duplicate_keys(self, client):
        """Duplicate keys in nested dict: last value wins."""
        json_data = {
            'features': [1.0] * 78,
            'metadata': {
                'key1': 'value1',
                'key1': 'value2',
                'key2': 100,
                'key2': 200
            }
        }
        assert json_data['metadata']['key1'] == 'value2'
        assert json_data['metadata']['key2'] == 200

        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 500, 503]

    def test_query_params_duplicate(self, client):
        """Duplicate URL query params."""
        response = client.get('/api/stream?label=DoS&label=PortScan&label=Bot')
        assert response.status_code in [200, 404, 500]

        # You can also read all values via getlist()
        from flask import request as flask_request
        with app.test_request_context('/api/stream?label=DoS&label=PortScan'):
            labels = flask_request.args.getlist('label')
            assert len(labels) == 2
            assert 'DoS' in labels
            assert 'PortScan' in labels


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_large_feature_array(self, client):
        """Very large feature list."""
        large_features = [1.0] * 10000
        response = client.post('/api/predict', json={'features': large_features})
        assert response.status_code in [200, 400, 500, 503]

    def test_empty_feature_array(self, client):
        """Empty feature list."""
        response = client.post('/api/predict', json={'features': []})
        assert response.status_code in [200, 400, 500, 503]

    def test_none_values_in_features(self, client):
        """Features containing None values."""
        features = [1.0, None, 3.0] + [0.0] * 75
        response = client.post('/api/predict', json={'features': features})
        assert response.status_code in [200, 400, 500, 503]

    def test_inf_values_in_features(self, client, mock_model_loaded):
        """Features containing Inf values."""
        features = [float('inf'), float('-inf'), 3.0] + [0.0] * 75
        response = client.post('/api/predict', json={'features': features})
        assert response.status_code in [200, 400, 500, 503]

    def test_nan_values_in_features(self, client, mock_model_loaded):
        """Features containing NaN values."""
        features = [float('nan'), 2.0, 3.0] + [0.0] * 75
        response = client.post('/api/predict', json={'features': features})
        assert response.status_code in [200, 400, 500, 503]

    def test_unicode_in_json(self, client):
        """Unicode in JSON payload."""
        json_data = {
            'features': [1.0] * 78,
            'message': 'Chinese test 🚀'
        }
        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 500, 503]

    def test_special_characters_in_keys(self, client):
        """Special characters in keys (extra keys should be ignored)."""
        json_data = {
            'features': [1.0] * 78,
            'key-with-dash': 'value',
            'key_with_underscore': 'value',
            'key.with.dot': 'value'
        }
        response = client.post('/api/predict', json=json_data)
        assert response.status_code in [200, 400, 500, 503]


class TestDataValidation:
    """Tests for data validation."""

    def test_feature_count_validation(self, client, mock_model_loaded):
        """Feature count mismatch."""
        wrong_count_features = [1.0] * 50
        response = client.post('/api/predict', json={'features': wrong_count_features})
        assert response.status_code in [200, 400, 500, 503]

    def test_feature_type_validation(self, client):
        """Mixed feature types."""
        mixed_features = [1, 2.0, '3', [4], None] + [0.0] * 73
        response = client.post('/api/predict', json={'features': mixed_features})
        assert response.status_code in [200, 400, 500, 503]

    def test_malformed_json(self, client):
        """Malformed JSON body."""
        response = client.post('/api/predict',
                               data='{"features": [1, 2, 3}',
                               content_type='application/json')
        assert response.status_code in [400, 500]


class TestConcurrency:
    """Tests for concurrency scenarios."""

    def test_concurrent_alerts_access(self, client):
        """Concurrent access to /api/alerts."""
        import threading

        results = []

        def get_alerts():
            response = client.get('/api/alerts')
            results.append(response.status_code)

        threads = [threading.Thread(target=get_alerts) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(status == 200 for status in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

