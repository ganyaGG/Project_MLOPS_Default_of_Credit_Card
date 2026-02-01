#!/bin/bash

# Blue-Green деплоймент стратегия

set -e

BLUE="credit-scoring-blue"
GREEN="credit-scoring-green"
NAMESPACE="production"
TRAFFIC_SPLIT="50"  # Начальное разделение трафика

# Проверяем текущую активную версию
CURRENT=$(kubectl get svc credit-scoring-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}')
echo "Текущая активная версия: $CURRENT"

# Определяем новую версию
if [ "$CURRENT" == "blue" ]; then
    NEW="green"
    OLD="blue"
else
    NEW="blue"
    OLD="green"
fi

echo "Развертывание новой версии: $NEW"

# Деплой новой версии
kubectl apply -f deployment/kubernetes/deployment-$NEW.yaml -n $NAMESPACE

# Ждем готовности
echo "Ожидание готовности новой версии..."
kubectl rollout status deployment/credit-scoring-$NEW -n $NAMESPACE --timeout=300s

# Начальное тестирование новой версии
echo "Запуск тестов новой версии..."
./scripts/test-new-version.sh $NEW

# Постепенное переключение трафика
echo "Начало переключения трафика..."

# Обновляем сервис для маршрутизации трафика
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: credit-scoring-service
  namespace: $NAMESPACE
spec:
  selector:
    app: credit-scoring
    version: $NEW
  ports:
  - port: 80
    targetPort: 8000
EOF

echo "Трафик переключен на версию: $NEW"

# Мониторинг метрик после переключения
echo "Мониторинг метрик..."
sleep 60
./scripts/check-metrics.sh

# Удаление старой версии (опционально)
echo "Удаление старой версии: $OLD"
kubectl delete deployment credit-scoring-$OLD -n $NAMESPACE

echo "Blue-Green деплоймент успешно завершен!"