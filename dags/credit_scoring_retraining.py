from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
import pendulum

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

def check_data_drift(**context):
    """Проверка дрифта данных"""
    import pandas as pd
    from monitoring.drift.drift_monitoring import DriftMonitor
    
    # Загрузка последних данных
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_data_path = f"/data/monitoring/daily_{current_date}.csv"
    
    # Инициализация монитора
    monitor = DriftMonitor(
        reference_data_path="/data/processed/train.csv"
    )
    
    # Проверка дрифта
    drift_result = monitor.detect_data_drift(
        pd.read_csv(current_data_path)
    )
    
    # Передача результата в следующий таск
    context['ti'].xcom_push(key='data_drift', value=drift_result)
    
    return drift_result['dataset_drift']

def check_concept_drift(**context):
    """Проверка концептуального дрифта"""
    import mlflow
    from sklearn.metrics import accuracy_score
    
    # Загрузка текущих метрик
    mlflow.set_tracking_uri(Variable.get("mlflow_tracking_uri"))
    
    # Получение последних метрик из MLflow
    # (упрощенная реализация)
    current_accuracy = 0.82  # В реальности получаем из мониторинга
    
    # Получение эталонных метрик
    reference_accuracy = 0.85
    
    concept_drift = (reference_accuracy - current_accuracy) > 0.05
    
    context['ti'].xcom_push(key='concept_drift', value=concept_drift)
    
    return concept_drift

def should_retrain(**context):
    """Определение необходимости переобучения"""
    ti = context['ti']
    
    data_drift = ti.xcom_pull(task_ids='check_data_drift')
    concept_drift = ti.xcom_pull(task_ids='check_concept_drift')
    
    should_retrain = data_drift or concept_drift
    
    context['ti'].xcom_push(key='should_retrain', value=should_retrain)
    
    if should_retrain:
        print("Retraining triggered due to:")
        if data_drift:
            print("  - Data drift detected")
        if concept_drift:
            print("  - Concept drift detected")
    else:
        print("No retraining needed")
    
    return should_retrain

def on_retraining_success(context):
    """Callback при успешном переобучении"""
    slack_msg = """
    :white_check_mark: *Model Retraining Successful*
    
    *Pipeline:* Credit Scoring Retraining
    *Timestamp:* {execution_date}
    *Duration:* {duration}
    
    Model successfully retrained and deployed.
    """.format(
        execution_date=context.get('execution_date'),
        duration=context.get('duration')
    )
    
    slack_alert = SlackWebhookOperator(
        task_id='slack_success',
        http_conn_id='slack_connection',
        message=slack_msg,
        username='airflow'
    )
    
    return slack_alert.execute(context=context)

def on_retraining_failure(context):
    """Callback при ошибке переобучения"""
    slack_msg = """
    :x: *Model Retraining Failed*
    
    *Pipeline:* Credit Scoring Retraining
    *Task:* {task}
    *Timestamp:* {execution_date}
    *Exception:* {exception}
    
    Please check the logs for details.
    """.format(
        task=context.get('task_instance').task_id,
        execution_date=context.get('execution_date'),
        exception=context.get('exception')
    )
    
    slack_alert = SlackWebhookOperator(
        task_id='slack_failure',
        http_conn_id='slack_connection',
        message=slack_msg,
        username='airflow'
    )
    
    return slack_alert.execute(context=context)

with DAG(
    'credit_scoring_retraining',
    default_args=default_args,
    description='Automated retraining pipeline for credit scoring model',
    schedule_interval='0 2 * * *',  # Ежедневно в 2 AM
    catchup=False,
    tags=['mlops', 'retraining', 'credit-scoring'],
    on_success_callback=on_retraining_success,
    on_failure_callback=on_retraining_failure
) as dag:
    
    start_pipeline = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Starting credit scoring retraining pipeline"'
    )
    
    check_data_quality = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True
    )
    
    check_drift_data = PythonOperator(
        task_id='check_data_drift',
        python_callable=check_data_drift,
        provide_context=True
    )
    
    check_drift_concept = PythonOperator(
        task_id='check_concept_drift',
        python_callable=check_concept_drift,
        provide_context=True
    )
    
    decide_retraining = PythonOperator(
        task_id='should_retrain',
        python_callable=should_retrain,
        provide_context=True
    )
    
    retrain_model = KubernetesPodOperator(
        task_id='retrain_model',
        namespace='ml-training',
        image='{{ var.value.training_image }}:latest',
        cmds=['python', '/app/train.py'],
        arguments=[
            '--data-path', '/data/processed/latest.csv',
            '--output-path', '/models/new_model.onnx',
            '--mlflow-uri', '{{ var.value.mlflow_tracking_uri }}'
        ],
        name='retrain-model-pod',
        is_delete_operator_pod=True,
        get_logs=True,
        env_vars={
            'MLFLOW_TRACKING_URI': Variable.get('mlflow_tracking_uri'),
            'DVC_REMOTE_URL': Variable.get('dvc_remote_url'),
            'AWS_ACCESS_KEY_ID': Variable.get('aws_access_key_id'),
            'AWS_SECRET_ACCESS_KEY': Variable.get('aws_secret_access_key')
        },
        resources={
            'request_memory': '4Gi',
            'request_cpu': '2000m',
            'limit_memory': '8Gi',
            'limit_cpu': '4000m'
        }
    )
    
    validate_model = KubernetesPodOperator(
        task_id='validate_new_model',
        namespace='ml-training',
        image='{{ var.value.validation_image }}:latest',
        cmds=['python', '/app/validate.py'],
        arguments=[
            '--model-path', '/models/new_model.onnx',
            '--test-data', '/data/processed/test.csv'
        ],
        name='validate-model-pod',
        is_delete_operator_pod=True,
        get_logs=True
    )
    
    deploy_canary = KubernetesPodOperator(
        task_id='deploy_canary',
        namespace='default',
        image='{{ var.value.deployment_image }}:latest',
        cmds=['python', '/app/deploy.py'],
        arguments=[
            '--strategy', 'canary',
            '--model-path', '/models/new_model.onnx',
            '--traffic-percentage', '10'
        ],
        name='deploy-canary-pod',
        is_delete_operator_pod=True,
        get_logs=True
    )
    
    monitor_canary = PythonOperator(
        task_id='monitor_canary',
        python_callable=monitor_canary_deployment,
        provide_context=True
    )
    
    promote_to_production = KubernetesPodOperator(
        task_id='promote_to_production',
        namespace='default',
        image='{{ var.value.deployment_image }}:latest',
        cmds=['python', '/app/deploy.py'],
        arguments=[
            '--strategy', 'full',
            '--model-path', '/models/new_model.onnx'
        ],
        name='promote-production-pod',
        is_delete_operator_pod=True,
        get_logs=True
    )
    
    update_model_registry = PythonOperator(
        task_id='update_model_registry',
        python_callable=update_model_in_registry,
        provide_context=True
    )
    
    cleanup_resources = BashOperator(
        task_id='cleanup_resources',
        bash_command='echo "Cleaning up temporary resources"'
    )
    
    # Определение зависимостей
    start_pipeline >> check_data_quality >> [check_drift_data, check_drift_concept]
    
    [check_drift_data, check_drift_concept] >> decide_retraining
    
    decide_retraining >> retrain_model >> validate_model >> deploy_canary
    
    deploy_canary >> monitor_canary >> promote_to_production
    
    promote_to_production >> update_model_registry >> cleanup_resources
    
    # Условное выполнение
    from airflow.operators.empty import EmptyOperator
    
    skip_retraining = EmptyOperator(
        task_id='skip_retraining',
        trigger_rule='all_failed'
    )
    
    decide_retraining >> skip_retraining >> cleanup_resources