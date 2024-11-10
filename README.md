# Подбор кандидатов на вакансию по типу личности

## Решение команды [RASCAR] O2
### Состав и роли:
Артём Гончаров - Team Lead, Full-stack, NLP\
Андрей Алёхин - CV, ML\
Игорь Карташов - CV, ML\
Артём Гуйван - Audio, NLP\
Александр Смирнов - Consulting

# Проблематика

Современный рынок труда сталкивается с многочисленными вызовами, связанными с адаптацией к
быстро меняющимся условиям экономики и технологическим преобразованиям. Многие соискатели и
работодатели сталкиваются с проблемами несоответствия личности кандидата специальности, что
приводит к неэффективному использованию человеческого капитала.
В связи с этим возникает необходимость в создании систем, которые могут более точно
предсказывать тип личности и сопоставлять его с требованиям по психотипу к вакансии.
Факты, свидетельствующие о значимости проблемы:
- Несоответствие личности кандидата к специальности: По данным многих исследований,
значительная часть соискателей не соответствует требованиям работодателей. Это ведет к
“текучести кадров” и происходит повышение стоимости поиска “стабильного” кандидата.
- Неопределенность кандидатов в карьерном пути: Многие кандидаты не знают какая им профессия
больше всего подойдет согласно их типу личности. Это ведет к долгому поиску работы.
- Большие данные и AI: Современные технологии анализа данных и искусственного интеллекта
предоставляют уникальные возможности для улучшения процессов найма, но они часто не
полностью используются.
Ожидаемые результаты:
- Повышение точности рекомендаций: Система будет предоставлять более подходящих кандидатов
под определенную вакансию по их типу личности.
- Улучшение взаимодействия на рынке труда: Работодатели смогут быстрее находить подходящих
кандидатов, а соискатели – получат рекомендации, на какие специальности им стоит обратить
внимание.
- Адаптивность и актуальность: Система будет постоянно обновляться и адаптироваться к
изменениям на рынке труда, обеспечивая актуальность своих рекомендаций и предсказаний.

# Постановка задачи

Главная задача:
Создание системы, способной по видеоматериалу подбирать кандидатов согласно их типу личности
под определенные вакансии.

# Описание решения

Представляем наше инновационное решение для подбора кандидатов на основе типов личности. Наш продукт включает удобный веб-интерфейс как для соискателей, так и для работодателей.

Для соискателей мы разработали страницу, где можно загрузить видео-визитку и получить вероятностные оценки по моделям OCEAN и MBTI. Для работодателей доступен интерфейс, позволяющий загружать большие объемы видео, автоматически получать оценки кандидатов по шкалам OCEAN и MBTI, а также гибко настраивать параметры поиска с помощью ползунков для точного подбора кандидатов.

Мы используем передовые технологии, такие как BERT, Whisper, Pandas, Flask, Librosa, Torchaudio, Torch, Transformers и SciPy. Уникальность нашего решения заключается в комплексной предобработке данных и анализе аудио, текста и видео, что позволяет максимально точно оценить личностные черты.

# Киллерфичи

### Особенности нашего решения:

- Индивидуальные характеристики и рекомендации для кандидатов
- Использование различных доменов данных для предсказания
- Возможность преобразования в другие системы оценки типа личности (Cattell's 16PF, HEXACO)
- Решение использует только опен-сорс технологии
- Решение не требует постоянного подключения к сети
- Все модели можно использовать в бизнесовых задачах
- Чат-бот психолог по оценки типа личности

# Масштабируемость

### Планы на развитие проекта в будущем:

- Удобный сервис для работодателей и кандидатов
- Высокая оценка модели
- Измеримость качества 6-ю релевантными метриками
- Продуманы все сценарии использования системы
(пользователь и админ)
- 10 дополнительных фич для лучшего user experience

# Стек технологий

Мы используем такие технологии, как:
- flask, bootstrap
- stable-whisper, BERT, Pandas, Flask, Librosa, Torchaudio, Torch, Transformers, SciPy

# Run


## Virtual environment

Мы предлагаем вам использовать Python версии 3.11.

### Создание виртуального окружения
```
python -m venv venv
```
### Активация окружения

#### Windows:
```
venv\Scripts\activate
```
#### Linux/MacOS:
```
source venv/bin/activate
```

## Скачайте модели

```
wget https://huggingface.co/artemgoncarov/catboost_models/resolve/main/agreeableness_best_model.cbm
wget https://huggingface.co/artemgoncarov/catboost_models/resolve/main/conscientiousness_best_model.cbm
wget https://huggingface.co/artemgoncarov/catboost_models/resolve/main/extraversion_best_model.cbm
wget https://huggingface.co/artemgoncarov/catboost_models/resolve/main/neuroticism_best_model.cbm
wget https://huggingface.co/artemgoncarov/catboost_models/resolve/main/openness_best_model.cbm
```

### Установка библиотек

```
pip install -r requirements.txt
```

## Запуск

```
python app.py
```

Далее у вас будет доступен веб-интерфейс по локальному адресу http://127.0.0.1:1488


# Описание файлов

## app.py

Главный файл - запуск веб-интерфейса.

## get_text.py

Транскрибация видео в текст

## nlp_model.py

Файл с обученной BERT моделью и кодом для инференса.

## ocean2mbti.py

Файл для перевода из системы OCEAN в MBTI.

# Рекомендуем к подписке на ТГ-каналы нашего сообщества

https://t.me/rascar_ai \
https://t.me/i_am_artemid \
https://t.me/ml_with_artem
