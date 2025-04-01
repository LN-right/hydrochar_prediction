import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
from flask import Flask, request, jsonify, render_template
import os
from tqdm import tqdm
import itertools

app = Flask(__name__)


# 加载模型和标准化器
def load_models():
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_Y.pkl', 'rb') as f:
        scaler_Y = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler_X, scaler_Y, model


scaler_X, scaler_Y, model = load_models()


# 生成所有实验条件组合
def generate_conditions():
    temperatures = np.arange(120, 361, 20)  # 120, 140,..., 360℃
    residence_times = np.arange(30, 181, 30)  # 30, 60,..., 180min
    solid_contents = np.arange(5, 51, 5)  # 5, 10,..., 30%
    return list(itertools.product(temperatures, residence_times, solid_contents))


# 预测函数
def predict_functional_groups(input_data):
    # 计算摩尔比
    C = input_data['c']
    H = input_data['h']
    O = input_data['o']
    N = input_data['n']
    HC = (H / 1) / (C / 12)
    NC = (N / 14) / (C / 12)
    OC = (O / 16) / (C / 12)

    # 构造输入数组
    x_input = np.array([
        C, H, O, N, input_data['s'], input_data['ash'],
        input_data['temperature'], input_data['residence'], input_data['solid'],
        HC, NC, OC
    ]).reshape(1, -1)

    # 预测
    x_scaled = scaler_X.transform(x_input)
    y_pred = scaler_Y.inverse_transform(model.predict(x_scaled))[0]

    return {
        'amino_n': float(y_pred[0]),
        'pyrrolic_n': float(y_pred[1]),
        'pyridinic_n': float(y_pred[2]),
        'quaternary_n': float(y_pred[3])
    }


# 优化函数
def optimize_conditions(input_data, target_func):
    # 获取固定元素值
    elements = {
        'c': input_data['c'],
        'h': input_data['h'],
        'o': input_data['o'],
        'n': input_data['n'],
        's': input_data['s'],
        'ash': input_data['ash']
    }
    C, H, O, N = elements['c'], elements['h'], elements['o'], elements['n']

    # 生成所有实验条件组合
    conditions = generate_conditions()

    # 遍历所有条件进行预测
    results = []
    for T, RT, SC in tqdm(conditions, desc="Processing Conditions"):
        # 计算摩尔比
        HC = (H / 1) / (C / 12)
        NC = (N / 14) / (C / 12)
        OC = (O / 16) / (C / 12)

        # 构造输入数组
        X = np.array([C, H, O, N, elements['s'], elements['ash'],
                      T, RT, SC, HC, NC, OC]).reshape(1, -1)

        # 预测
        X_scaled = scaler_X.transform(X)
        y_pred = scaler_Y.inverse_transform(model.predict(X_scaled))[0]

        # 存储结果
        results.append({
            'temperature': int(T),  # 转换为Python int
            'residence_time': int(RT),  # 转换为Python int
            'solid_content': int(SC),  # 转换为Python int
            'amino_n': float(y_pred[0]),
            'pyrrolic_n': float(y_pred[1]),
            'pyridinic_n': float(y_pred[2]),
            'quaternary_n': float(y_pred[3])
        })

    # 按目标官能团排序
    target_key = target_func.lower().replace('-', '_')
    sorted_results = sorted(results, key=lambda x: x[target_key], reverse=True)[:10]

    return sorted_results


# 首页路由
@app.route('/')
def home():
    return render_template('index.html')


# 预测API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 验证输入
        required_fields = ['c', 'h', 'o', 'n', 's', 'ash', 'temperature', 'residence', 'solid']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Invalid or missing field: {field}'}), 400

        # 执行预测
        prediction = predict_functional_groups({
            'c': float(data['c']),
            'h': float(data['h']),
            'o': float(data['o']),
            'n': float(data['n']),
            's': float(data['s']),
            'ash': float(data['ash']),
            'temperature': float(data['temperature']),
            'residence': float(data['residence']),
            'solid': float(data['solid'])
        })

        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 优化API
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()

        # 验证输入
        required_fields = ['c', 'h', 'o', 'n', 's', 'ash']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Invalid or missing field: {field}'}), 400

        if 'target_func' not in data or data['target_func'] not in ['Amino-N', 'Pyrrolic-N', 'Pyridinic-N',
                                                                    'Quaternary-N']:
            return jsonify({'error': 'Invalid or missing target functional group'}), 400

        # 执行优化
        optimized = optimize_conditions({
            'c': float(data['c']),
            'h': float(data['h']),
            'o': float(data['o']),
            'n': float(data['n']),
            's': float(data['s']),
            'ash': float(data['ash'])
        }, data['target_func'])

        # 确保所有数据都是JSON可序列化的
        return jsonify({
            'results': optimized,
            'target_func': data['target_func']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)