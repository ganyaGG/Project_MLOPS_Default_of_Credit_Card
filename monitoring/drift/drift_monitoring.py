import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    DataQualityMetricsTable,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    RegressionQualityMetric
)
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import mlflow
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

class DriftMonitor:
    def __init__(self, reference_data_path, mlflow_tracking_uri="http://localhost:5000"):
        """
        Инициализация монитора дрифта
        
        Args:
            reference_data_path: путь к эталонным данным
            mlflow_tracking_uri: URI MLflow сервера
        """
        self.reference_data = pd.read_csv(reference_data_path)
        self.mlflow_client = MlflowClient(mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Определяем числовые и категориальные признаки
        self.numerical_features = self.reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        self.categorical_features = self.reference_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Исключаем целевую переменную
        if 'default_payment_next_month' in self.numerical_features:
            self.numerical_features.remove('default_payment_next_month')
        
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
    
    def detect_data_drift(self, current_data, timestamp=None):
        """
        Обнаружение дрифта данных
        
        Args:
            current_data: текущие данные для анализа
            timestamp: временная метка анализа
            
        Returns:
            dict: результаты анализа дрифта
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Создание отчета Evidently
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityMetricsTable()
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Извлечение результатов
        result = data_drift_report.as_dict()
        
        # Анализ дрифта для каждого признака
        drift_results = {}
        for feature in self.numerical_features + self.categorical_features:
            if feature in current_data.columns:
                column_report = Report(metrics=[
                    ColumnDriftMetric(column_name=feature)
                ])
                column_report.run(
                    reference_data=self.reference_data,
                    current_data=current_data
                )
                col_result = column_report.as_dict()
                
                drift_results[feature] = {
                    'drift_detected': col_result['metrics'][0]['result']['drift_detected'],
                    'drift_score': col_result['metrics'][0]['result']['drift_score'],
                    'current_distribution': col_result['metrics'][0]['result']['current']['distribution'],
                    'reference_distribution': col_result['metrics'][0]['result']['reference']['distribution']
                }
        
        # Сохранение отчета
        report_path = f"monitoring/reports/drift_report_{timestamp}.html"
        data_drift_report.save_html(report_path)
        
        # Сохранение в MLflow
        with mlflow.start_run(run_name=f"drift_detection_{timestamp}"):
            mlflow.log_artifact(report_path)
            
            # Логирование метрик дрифта
            dataset_drift = result['metrics'][0]['result']['dataset_drift']
            mlflow.log_metric("dataset_drift", float(dataset_drift))
            mlflow.log_metric("drifted_features", result['metrics'][0]['result']['number_of_drifted_columns'])
            
            for feature, metrics in drift_results.items():
                if metrics['drift_detected']:
                    mlflow.log_metric(f"drift_{feature}", metrics['drift_score'])
        
        return {
            'timestamp': timestamp,
            'dataset_drift': dataset_drift,
            'drifted_features_count': result['metrics'][0]['result']['number_of_drifted_columns'],
            'feature_drift_details': drift_results,
            'report_path': report_path
        }
    
    def detect_concept_drift(self, y_true, y_pred, reference_metrics):
        """
        Обнаружение концептуального дрифта
        
        Args:
            y_true: истинные значения
            y_pred: предсказанные значения
            reference_metrics: эталонные метрики производительности
            
        Returns:
            dict: результаты концептуального дрифта
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        concept_drift = False
        drift_details = {}
        
        for metric_name, current_value in current_metrics.items():
            reference_value = reference_metrics.get(metric_name, 0)
            
            # Проверка значительного падения производительности
            if reference_value > 0:
                performance_drop = (reference_value - current_value) / reference_value
                
                if performance_drop > 0.1:  # Более 10% падение
                    concept_drift = True
                    drift_details[metric_name] = {
                        'current': current_value,
                        'reference': reference_value,
                        'drop_percentage': performance_drop * 100
                    }
        
        return {
            'concept_drift_detected': concept_drift,
            'current_metrics': current_metrics,
            'reference_metrics': reference_metrics,
            'drift_details': drift_details
        }
    
    def generate_alert(self, drift_result, threshold=0.5):
        """
        Генерация алертов на основе дрифта
        
        Args:
            drift_result: результаты анализа дрифта
            threshold: порог для триггера алерта
            
        Returns:
            dict: информация об алерте
        """
        alerts = []
        
        # Проверка дрифта датасета
        if drift_result['dataset_drift']:
            alerts.append({
                'type': 'dataset_drift',
                'severity': 'critical',
                'message': f'Dataset drift detected! {drift_result["drifted_features_count"]} features drifted.',
                'timestamp': drift_result['timestamp']
            })
        
        # Проверка дрифта отдельных признаков
        for feature, details in drift_result['feature_drift_details'].items():
            if details['drift_detected'] and details['drift_score'] > threshold:
                alerts.append({
                    'type': 'feature_drift',
                    'severity': 'warning',
                    'feature': feature,
                    'drift_score': details['drift_score'],
                    'message': f'Significant drift detected in feature: {feature}',
                    'timestamp': drift_result['timestamp']
                })
        
        # Отправка алертов
        if alerts:
            self.send_alerts(alerts)
        
        return alerts
    
    def send_alerts(self, alerts):
        """Отправка алертов в различные системы"""
        # Отправка в Slack
        try:
            import requests
            
            webhook_url = "https://hooks.slack.com/services/your/webhook/url"
            
            for alert in alerts:
                message = {
                    "text": f"🚨 {alert['severity'].upper()} Alert: {alert['message']}",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*{alert['severity'].upper()} Alert*"
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Type:*\n{alert['type']}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Time:*\n{alert['timestamp']}"
                                }
                            ]
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Message:*\n{alert['message']}"
                            }
                        }
                    ]
                }
                
                requests.post(webhook_url, json=message)
                
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
        
        # Логирование алертов
        with open('monitoring/alerts/alerts.log', 'a') as f:
            for alert in alerts:
                f.write(json.dumps(alert) + '\n')
    
    def run_monitoring_pipeline(self, current_data_path, y_true=None, y_pred=None):
        """
        Запуск полного пайплайна мониторинга
        
        Args:
            current_data_path: путь к текущим данным
            y_true: истинные метки (опционально)
            y_pred: предсказания (опционально)
        """
        print("Starting drift monitoring pipeline...")
        
        # Загрузка текущих данных
        current_data = pd.read_csv(current_data_path)
        timestamp = datetime.now().isoformat()
        
        # 1. Обнаружение дрифта данных
        print("Step 1: Detecting data drift...")
        data_drift_result = self.detect_data_drift(current_data, timestamp)
        
        # 2. Обнаружение концептуального дрифта (если доступны метки)
        concept_drift_result = None
        if y_true is not None and y_pred is not None:
            print("Step 2: Detecting concept drift...")
            reference_metrics = {
                'accuracy': 0.82,
                'precision': 0.78,
                'recall': 0.75,
                'f1': 0.76
            }
            concept_drift_result = self.detect_concept_drift(y_true, y_pred, reference_metrics)
        
        # 3. Генерация алертов
        print("Step 3: Generating alerts...")
        alerts = self.generate_alert(data_drift_result)
        
        if concept_drift_result and concept_drift_result['concept_drift_detected']:
            alerts.append({
                'type': 'concept_drift',
                'severity': 'critical',
                'message': 'Concept drift detected! Model performance degraded.',
                'details': concept_drift_result['drift_details'],
                'timestamp': timestamp
            })
        
        # 4. Сохранение результатов
        print("Step 4: Saving results...")
        results = {
            'timestamp': timestamp,
            'data_drift': data_drift_result,
            'concept_drift': concept_drift_result,
            'alerts': alerts
        }
        
        results_file = f"monitoring/results/monitoring_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Monitoring pipeline completed. Results saved to {results_file}")
        
        return results

# Пример использования
if __name__ == "__main__":
    # Инициализация монитора
    monitor = DriftMonitor(
        reference_data_path="data/processed/train.csv",
        mlflow_tracking_uri="http://localhost:5000"
    )
    
    # Ежедневный мониторинг
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_data_path = f"data/monitoring/daily_{current_date}.csv"
    
    # Запуск мониторинга
    results = monitor.run_monitoring_pipeline(current_data_path)
    
    # Визуализация результатов
    monitor.visualize_results(results)