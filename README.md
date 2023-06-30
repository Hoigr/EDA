#### Данный проект по курсу Разработка ML сервиса: от идеи к прототипу https://stepik.org/course/176820/syllabus
### Файлы:
- **data.py** - Методы для препроцессинга данных
- **model.py** - Методы для обучения модели
- **main.py** - GUI обертка поекта
- **pred.csv** - костыли что бы работала загрузка модели и предсказаниие (да мне стыдно за это). Передача из streamlit данных для предсказания передаются через этот файл 
- **cat_model** - обученная модель для catboost (формат cbm)
- **st.csv** - Статистические данные по датасету(min, max, mean, median)

### Ссылки:
##### - Ноутбук 
https://colab.research.google.com/drive/15cqMYf4K624bgp6CTxSDcLTf-_BjLw8L?usp=sharing
#####  - Датасет 
https://drive.google.com/file/d/1wgUPYZTnUSMSNu85uqZrdFtjEDSltzV-/view?usp=sharing
##### - Предобученная модель
https://drive.google.com/file/d/1jclwnAyIaC-8lIr_OxfjAXZPpBL1wUk8/view?usp=sharing
##### - Статистические данные
https://drive.google.com/file/d/1-5Ruis707ve6YVfgqLMV40TR2CHkZrS6/view?usp=sharing

### Краткое описания работы над моделью:
- Загрузка данных
- Замена пропущенных данных на среднее арифметическое значение по соответствующему сттолбцу
- Поиск выбросов ианомалий (библиотека PYOD метод случайный лес)
- Разбиение  датасета на тренировочную и тестовую выборку предварительно убрав аномалии и выбросы
- Обучение модели методом градиентного бустинга (библиотека catboost)

#### Примичание.
К сожалению  не сумел подружить библиотеку  sketch (https://pypi.org/project/sketch/) со streamlit. В google colab работает  хорошо а при использовании streamlit все время вылазит ошибка (не хочет видить датасет)
