# crypten-linreg

Обучение линейной регрессии в CrypTen.

Скрипт генерации данных и открытое обучение линейной регрессии в pytorch в `notebooks/linreg-plain.ipynb`.

Для запуска:
1. Указать в `docker-compose.yml` в разделе `command` для каждого воркера нужную функцию из `src/tasks/mpc.py`.
2. Выполнить `docker compose up`.

После внесения изменений в код:
1. Остановить и удалить прерыдущие контейнеры: 
```bash
docker compose down
```

2. Запустить, пересобрав новые образы:
```bash
docker compose up --build
```