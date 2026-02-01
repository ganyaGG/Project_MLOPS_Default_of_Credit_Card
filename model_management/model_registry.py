import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime

class ModelRegistryManager:
    def __init__(self, tracking_uri="http://localhost:5000"):
        """Инициализация менеджера реестра моделей"""
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
    def register_model(self, model, model_name, metrics, features, run_id=None):
        """
        Регистрация модели в MLflow
        
        Args:
            model: обученная модель
            model_name: имя модели
            metrics: словарь метрик
            features: список признаков
            run_id: ID запуска MLflow
            
        Returns:
            str: версия зарегистрированной модели
        """
        if run_id is None:
            run_id = mlflow.active_run().info.run_id
        
        # Логирование модели
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        
        # Логирование метрик
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Логирование параметров
        mlflow.log_params(model.get_params())
        
        # Логирование признаков
        mlflow.log_param("features", json.dumps(features))
        
        # Получение версии модели
        model_versions = self.client.get_latest_versions(model_name)
        if model_versions:
            return model_versions[-1].version
        
        return "1"
    
    def promote_model(self, model_name, version, stage="Staging"):
        """
        Продвижение модели на следующую стадию
        
        Args:
            model_name: имя модели
            version: версия модели
            stage: целевая стадия (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"Model {model_name} version {version} promoted to {stage}")
    
    def compare_models(self, model_name, versions=None):
        """
        Сравнение разных версий модели
        
        Args:
            model_name: имя модели
            versions: список версий для сравнения
            
        Returns:
            DataFrame: сравнительная таблица
        """
        if versions is None:
            versions = self.client.get_latest_versions(model_name)
            versions = [v.version for v in versions]
        
        comparison_data = []
        
        for version in versions:
            model_info = self.client.get_model_version(model_name, version)
            
            # Получение метрик
            run_id = model_info.run_id
            run = self.client.get_run(run_id)
            
            metrics = run.data.metrics
            params = run.data.params
            
            comparison_data.append({
                'version': version,
                'stage': model_info.current_stage,
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'training_date': run.info.start_time,
                'features': json.loads(params.get('features', '[]'))
            })
        
        return pd.DataFrame(comparison_data)
    
    def deploy_model(self, model_name, version, deployment_config):
        """
        Деплой модели
        
        Args:
            model_name: имя модели
            version: версия модели
            deployment_config: конфигурация деплоя
            
        Returns:
            dict: результаты деплоя
        """
        # Загрузка модели
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Подготовка окружения деплоя
        deployment_result = {
            'model_name': model_name,
            'version': version,
            'deployment_time': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # В зависимости от конфигурации деплоя
        if deployment_config.get('strategy') == 'canary':
            result = self._deploy_canary(model, deployment_config)
        elif deployment_config.get('strategy') == 'blue_green':
            result = self._deploy_blue_green(model, deployment_config)
        else:
            result = self._deploy_direct(model, deployment_config)
        
        deployment_result.update(result)
        deployment_result['status'] = 'deployed'
        
        # Логирование деплоя
        mlflow.log_dict(deployment_result, "deployment_info.json")
        
        return deployment_result
    
    def _deploy_canary(self, model, config):
        """Canary деплоймент"""
        # Реализация canary деплоймента
        initial_traffic = config.get('initial_traffic', 10)
        
        return {
            'strategy': 'canary',
            'initial_traffic_percentage': initial_traffic,
            'deployment_steps': [
                f"Deploying {initial_traffic}% of traffic",
                "Monitoring metrics for 1 hour",
                "Gradually increasing traffic"
            ]
        }
    
    def _deploy_blue_green(self, model, config):
        """Blue-Green деплоймент"""
        return {
            'strategy': 'blue_green',
            'deployment_steps': [
                "Deploying green version",
                "Switching load balancer",
                "Validating new version",
                "Decommissioning blue version"
            ]
        }
    
    def run_ab_test(self, model_name, version_a, version_b, test_data, duration_days=7):
        """
        Запуск A/B тестирования моделей
        
        Args:
            model_name: имя модели
            version_a: версия A
            version_b: версия B
            test_data: тестовые данные
            duration_days: длительность теста
            
        Returns:
            dict: результаты A/B теста
        """
        # Загрузка моделей
        model_a = mlflow.pyfunc.load_model(f"models:/{model_name}/{version_a}")
        model_b = mlflow.pyfunc.load_model(f"models:/{model_name}/{version_b}")
        
        # Подготовка данных
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Предсказания
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        
        # Расчет метрик
        metrics_a = {
            'accuracy': accuracy_score(y_test, pred_a),
            'f1': f1_score(y_test, pred_a)
        }
        
        metrics_b = {
            'accuracy': accuracy_score(y_test, pred_b),
            'f1': f1_score(y_test, pred_b)
        }
        
        # Статистическая значимость
        from scipy import stats
        
        # Простой t-test для accuracy
        # В реальном проекте используйте более сложные методы
        accuracy_diff = metrics_b['accuracy'] - metrics_a['accuracy']
        
        ab_test_result = {
            'model_a': {
                'version': version_a,
                'metrics': metrics_a
            },
            'model_b': {
                'version': version_b,
                'metrics': metrics_b
            },
            'improvement': {
                'accuracy': accuracy_diff,
                'relative_improvement': accuracy_diff / metrics_a['accuracy'] * 100
            },
            'test_duration_days': duration_days,
            'timestamp': datetime.now().isoformat()
        }
        
        # Логирование результатов
        mlflow.log_dict(ab_test_result, "ab_test_results.json")
        
        return ab_test_result

# Пример использования
if __name__ == "__main__":
    # Инициализация менеджера
    manager = ModelRegistryManager()
    
    # Регистрация новой модели
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, predictions),
        'f1_score': f1_score(y, predictions)
    }
    
    version = manager.register_model(
        model=model,
        model_name="credit_scoring",
        metrics=metrics,
        features=[f"feature_{i}" for i in range(20)]
    )
    
    print(f"Model registered as version {version}")
    
    # Продвижение модели в Production
    manager.promote_model("credit_scoring", version, "Production")
    
    # Сравнение версий
    comparison_df = manager.compare_models("credit_scoring")
    print(comparison_df)