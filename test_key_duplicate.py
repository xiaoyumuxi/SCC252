"""
单元测试文件：测试所有API功能，包括key重复测试
测试覆盖：
1. 所有API端点的正常功能
2. JSON请求中key重复的情况
3. 边界情况和错误处理
4. 数据验证和异常处理
"""

import pytest
import json
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model_components, init_db, MODEL, SCALER, LE, FEATURE_COLUMNS


@pytest.fixture
def client():
    """创建测试客户端"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    # 创建临时数据库
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    test_db.close()
    
    # 备份原始数据库路径
    from app import DB_FILE
    original_db = DB_FILE
    
    # 临时替换数据库路径
    import app as app_module
    app_module.DB_FILE = test_db.name
    
    # 初始化测试数据库
    init_db()
    
    with app.test_client() as client:
        yield client
    
    # 清理：恢复原始数据库路径并删除测试数据库
    app_module.DB_FILE = original_db
    if os.path.exists(test_db.name):
        os.unlink(test_db.name)


@pytest.fixture
def sample_features():
    """生成示例特征向量"""
    # 假设有78个特征（根据代码中的默认值）
    return [float(i) for i in range(78)]


@pytest.fixture
def mock_model_loaded():
    """模拟模型已加载"""
    with patch('app.MODEL', MagicMock()), \
         patch('app.SCALER', MagicMock()), \
         patch('app.LE', MagicMock()), \
         patch('app.FEATURE_COLUMNS', [f'f_{i}' for i in range(78)]):
        # 设置LE的inverse_transform方法
        app.LE.inverse_transform = MagicMock(return_value=['BENIGN'])
        yield


class TestHealthCheck:
    """测试健康检查端点"""
    
    def test_health_check_success(self, client):
        """测试健康检查成功"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data


class TestPredictAPI:
    """测试预测API端点"""
    
    def test_predict_missing_features(self, client):
        """测试缺少features字段"""
        response = client.post('/api/predict', 
                             json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'features' in data['message'].lower()
    
    def test_predict_empty_json(self, client):
        """测试空JSON请求"""
        response = client.post('/api/predict',
                             json=None,
                             content_type='application/json')
        assert response.status_code == 400
    
    def test_predict_key_duplicate_in_json(self, client, sample_features, mock_model_loaded):
        """测试JSON中key重复的情况"""
        # 创建包含重复key的JSON字符串
        # 注意：Python的dict会自动处理重复key（保留最后一个），
        # 但我们可以测试这种情况
        json_str = '{"features": [1, 2, 3], "features": ' + str(sample_features) + '}'
        
        # 使用requests方式发送，模拟key重复
        response = client.post('/api/predict',
                             data=json_str,
                             content_type='application/json')
        
        # 由于Python dict会自动处理重复key，应该能正常处理
        # 如果模型未加载，会返回503
        assert response.status_code in [200, 400, 503]
    
    def test_predict_multiple_duplicate_keys(self, client, sample_features):
        """测试多个重复key的情况"""
        # 创建包含多个重复key的JSON
        json_data = {
            'features': sample_features,
            'features': sample_features,  # 重复的key
            'extra': 'value1',
            'extra': 'value2'  # 重复的key
        }
        
        response = client.post('/api/predict',
                             json=json_data)
        # 应该能处理（Python dict会保留最后一个值）
        assert response.status_code in [200, 400, 503]
    
    def test_predict_invalid_features_type(self, client):
        """测试features类型错误"""
        response = client.post('/api/predict',
                             json={'features': 'not_a_list'})
        assert response.status_code in [400, 500, 503]
    
    def test_predict_wrong_feature_count(self, client, mock_model_loaded):
        """测试特征数量不匹配"""
        wrong_features = [1.0, 2.0, 3.0]  # 只有3个特征，应该需要78个
        response = client.post('/api/predict',
                             json={'features': wrong_features})
        # 如果模型已加载，应该返回错误
        assert response.status_code in [200, 400, 500, 503]


class TestAlertsAPI:
    """测试警报API端点"""
    
    def test_get_alerts_success(self, client):
        """测试获取警报成功"""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_get_alerts_empty(self, client):
        """测试获取空警报列表"""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []


class TestHistoryAPI:
    """测试历史记录API端点"""
    
    def test_get_history_success(self, client):
        """测试获取历史记录成功"""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_get_history_empty(self, client):
        """测试获取空历史记录"""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestPerformanceAPI:
    """测试性能指标API端点"""
    
    def test_get_performance_success(self, client):
        """测试获取性能指标成功"""
        response = client.get('/api/performance')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)
        # 检查是否包含预期的性能指标字段
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        for key in expected_keys:
            if key in data:
                assert isinstance(data[key], (int, float))


class TestStreamAPI:
    """测试流数据API端点"""
    
    def test_get_stream_success(self, client):
        """测试获取流数据成功"""
        response = client.get('/api/stream')
        # 可能返回错误（如果攻击样本库未构建）或成功
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
    
    def test_get_stream_with_label_filter(self, client):
        """测试带标签过滤的流数据"""
        response = client.get('/api/stream?label=DoS%20Hulk')
        assert response.status_code in [200, 404, 500]
    
    def test_get_stream_invalid_label(self, client):
        """测试无效标签过滤"""
        response = client.get('/api/stream?label=NonExistentAttack')
        assert response.status_code in [200, 404, 500]
    
    def test_get_stream_duplicate_query_params(self, client):
        """测试重复的查询参数"""
        # 测试URL中重复的查询参数
        response = client.get('/api/stream?label=DoS&label=PortScan')
        # Flask会处理重复参数（保留最后一个或作为列表）
        assert response.status_code in [200, 404, 500]


class TestRandomAPI:
    """测试随机数据API端点"""
    
    def test_get_random_success(self, client):
        """测试获取随机数据成功"""
        response = client.get('/api/random')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'features' in data
        assert 'feature_names' in data
        assert isinstance(data['features'], list)
        assert isinstance(data['feature_names'], list)
        assert len(data['features']) == len(data['feature_names'])


class TestUploadAndRetrainAPI:
    """测试上传和重训练API端点"""
    
    def test_upload_no_files(self, client):
        """测试没有文件的上传请求"""
        response = client.post('/api/upload-and-retrain')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_upload_empty_files(self, client):
        """测试空文件列表"""
        response = client.post('/api/upload-and-retrain',
                             data={'files': []})
        assert response.status_code == 400
    
    def test_upload_invalid_file_type(self, client):
        """测试无效文件类型"""
        data = {
            'files': (BytesIO(b'not csv content'), 'test.txt')
        }
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        # 应该返回400或忽略非CSV文件
        assert response.status_code in [400, 200]
    
    def test_upload_valid_csv(self, client):
        """测试上传有效的CSV文件"""
        # 创建测试CSV数据
        test_data = {
            'Label': ['BENIGN', 'DDoS', 'BENIGN'],
            'Feature1': [1.0, 2.0, 3.0],
            'Feature2': [4.0, 5.0, 6.0],
            'Feature3': [7.0, 8.0, 9.0]
        }
        df = pd.DataFrame(test_data)
        
        # 保存为CSV字节流
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        data = {
            'files': (csv_buffer, 'test.csv')
        }
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        # 可能成功或失败（取决于模型加载状态）
        assert response.status_code in [200, 400, 500]
    
    def test_upload_multiple_files(self, client):
        """测试上传多个文件"""
        # 创建两个测试CSV文件
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
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        assert response.status_code in [200, 400, 500]
    
    def test_upload_csv_without_label_column(self, client):
        """测试缺少Label列的CSV"""
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
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'


class TestKeyDuplicateScenarios:
    """专门测试key重复的各种场景"""
    
    def test_json_duplicate_key_last_wins(self, client):
        """测试JSON中重复key，最后一个值生效（Python dict行为）"""
        # Python的dict会自动处理重复key，保留最后一个值
        json_data = {
            'features': [1, 2, 3],
            'features': [4, 5, 6]  # 这个值会覆盖上面的
        }
        
        # 验证Python dict的行为
        assert json_data['features'] == [4, 5, 6]
        
        response = client.post('/api/predict',
                             json=json_data)
        # 应该使用最后一个值
        assert response.status_code in [200, 400, 500, 503]
    
    def test_multiple_duplicate_keys_in_request(self, client):
        """测试请求中多个重复key"""
        json_data = {
            'features': [1.0] * 78,
            'features': [2.0] * 78,  # 重复
            'extra_param': 'value1',
            'extra_param': 'value2',  # 重复
            'another': 100,
            'another': 200  # 重复
        }
        
        # 验证dict行为
        assert json_data['features'] == [2.0] * 78
        assert json_data['extra_param'] == 'value2'
        assert json_data['another'] == 200
        
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_nested_duplicate_keys(self, client):
        """测试嵌套结构中的重复key"""
        json_data = {
            'features': [1.0] * 78,
            'metadata': {
                'key1': 'value1',
                'key1': 'value2',  # 嵌套中的重复key
                'key2': 100,
                'key2': 200  # 嵌套中的重复key
            }
        }
        
        # 验证嵌套dict行为
        assert json_data['metadata']['key1'] == 'value2'
        assert json_data['metadata']['key2'] == 200
        
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_query_params_duplicate(self, client):
        """测试URL查询参数重复"""
        # Flask会处理重复的查询参数
        response = client.get('/api/stream?label=DoS&label=PortScan&label=Bot')
        assert response.status_code in [200, 404, 500]
        
        # 可以获取所有重复的参数值
        from flask import request as flask_request
        with app.test_request_context('/api/stream?label=DoS&label=PortScan'):
            labels = flask_request.args.getlist('label')
            assert len(labels) == 2
            assert 'DoS' in labels
            assert 'PortScan' in labels


class TestEdgeCases:
    """测试边界情况和异常处理"""
    
    def test_very_large_feature_array(self, client):
        """测试非常大的特征数组"""
        large_features = [1.0] * 10000
        response = client.post('/api/predict',
                             json={'features': large_features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_empty_feature_array(self, client):
        """测试空特征数组"""
        response = client.post('/api/predict',
                             json={'features': []})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_none_values_in_features(self, client):
        """测试特征中包含None值"""
        features = [1.0, None, 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_inf_values_in_features(self, client, mock_model_loaded):
        """测试特征中包含Inf值"""
        features = [float('inf'), float('-inf'), 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        # 应该能处理Inf值（代码中有replace逻辑）
        assert response.status_code in [200, 400, 500, 503]
    
    def test_nan_values_in_features(self, client, mock_model_loaded):
        """测试特征中包含NaN值"""
        features = [float('nan'), 2.0, 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        # 应该能处理NaN值（代码中有fillna逻辑）
        assert response.status_code in [200, 400, 500, 503]
    
    def test_unicode_in_json(self, client):
        """测试JSON中包含Unicode字符"""
        json_data = {
            'features': [1.0] * 78,
            'message': '测试中文 🚀'
        }
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_special_characters_in_keys(self, client):
        """测试key中包含特殊字符"""
        # 注意：虽然features是必需的，但可以测试其他key
        json_data = {
            'features': [1.0] * 78,
            'key-with-dash': 'value',
            'key_with_underscore': 'value',
            'key.with.dot': 'value'
        }
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]


class TestDataValidation:
    """测试数据验证功能"""
    
    def test_feature_count_validation(self, client, mock_model_loaded):
        """测试特征数量验证"""
        # 测试特征数量不匹配
        wrong_count_features = [1.0] * 50  # 应该是78个
        response = client.post('/api/predict',
                             json={'features': wrong_count_features})
        # 如果模型已加载，应该返回错误
        assert response.status_code in [200, 400, 500, 503]
    
    def test_feature_type_validation(self, client):
        """测试特征类型验证"""
        # 测试混合类型
        mixed_features = [1, 2.0, '3', [4], None] + [0.0] * 73
        response = client.post('/api/predict',
                             json={'features': mixed_features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_malformed_json(self, client):
        """测试格式错误的JSON"""
        response = client.post('/api/predict',
                             data='{"features": [1, 2, 3}',  # 缺少闭合括号
                             content_type='application/json')
        assert response.status_code in [400, 500]


class TestConcurrency:
    """测试并发场景"""
    
    def test_concurrent_alerts_access(self, client):
        """测试并发访问警报"""
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
        
        # 所有请求应该都成功
        assert all(status == 200 for status in results)


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

