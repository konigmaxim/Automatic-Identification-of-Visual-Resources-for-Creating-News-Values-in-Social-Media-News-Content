# Automatic-Identification-of-Visual-Resources-for-Creating-News-Values-in-Social-Media-News-Content
Данный репозиторий используется для хранения и демонстрации разработки экспериментальной модели автоматической классификации визуальных новостных ценностей на основе DNVA и ее апробации на визуальном контенте для социальных сетей. Он содержит код, данные и обученную модель для курсовой работы «Автоматическая идентификация визуальных ресурсов для создания новостных ценностей в новостном контенте социальных сетей» (Национальный исследовательский университет «Высшая школа экономики», факультет гуманитарных наук, направление «Фундаментальная и компьютерная лингвистика», 2026).

## Краткое описание проекта
Цель работы — разработать систему автоматической классификации визуальных новостных ценностей (Negativity, Positivity, Personalization, Superlativeness, Eliteness, Aesthetic Appeal) на основе теории дискурсивного анализа (Bednarek, Caple 2017). Модель (ResNet-50) обучена на  выборке из 200 изображений, размеченных по шести ценностям. Классификация multi‑label: одно изображение может нести несколько ценностей одновременно.

## Структура репозитория

```
├── README.md               # описание проекта и инструкции
├── dataset.csv             # таблица с метками и ссылками на изображения
├── images/                 # папка с изображениями (image1.png … imageN.png)
├── train_model.py          # основной скрипт для обучения, кросс‑валидации и оценки
└── requirements.txt        # зависимости Python
```

## Требования

- Python 3.8+
- PyTorch (желательно с поддержкой CUDA)
- torchvision
- pandas, numpy, scikit‑learn, Pillow, tqdm, openpyxl

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Данные

Файл `dataset.csv` содержит 9 столбцов:

1. `Source` – источник (издание)
2. `Link` – ссылка на публикацию
3. `Headline` – заголовок новости
4. `Image_Pos` – Позитивность (0/1)
5. `Image_Neg` – Негативность (0/1)
6. `Image_Pers` – Персонализация (0/1)
7. `Image_Super` – Суперлативность (0/1)
8. `Image_Elite` – Элитность (0/1)
9. `Image_AA` – Эстетическая привлекательность (0/1)

Изображения лежат в папке `images/`. Всего 200 изображений в формате PNG. Они были собраны с новостных сайтов и из социальных сетей (период: январь–май 2026 г.). Разметка выполнена вручную согласно критериям DNVA.

## Обучение модели

Скрипт `train_model.py` выполняет:

- Загрузку данных из Excel (dataset.csv) и проверку наличия изображений.
- Предобработку: Resize (224×224), аугментации (RandomHorizontalFlip, ColorJitter) для обучающей выборки.
- Пятикратную стратифицированную кросс‑валидацию с метрикой macro F1.
- Финальное обучение на 80% данных с подбором оптимальных порогов уверенности для каждого класса.
- Сохранение модели в `final_model_balanced.pth`.

Для запуска обучения разместите репозиторий в своей среде и выполните:

```bash
python train_model.py
```

Убедитесь, что пути к папке `images` и файлу `dataset.csv` корректны (при необходимости измените переменные `base_directory` и `path_to_dataset` в коде).

## Результаты
По итогам кросс‑валидации среднее macro F1 составило **0.66**. После подбора порогов качество улучшилось до macro F1 **0.72** (на тестовой выборке). Классы, которые модель путает чаще всего, и полный отчёт приведены в тексте курсовой работы.

## Использование готовой модели (inference)

Пример загрузки модели и предсказания для одного изображения:

```python
import torch
from torchvision import transforms, models
from PIL import Image

device = torch.device('cpu')
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.fc.in_features, 6)
)
model.load_state_dict(torch.load('final_model_balanced.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs).squeeze()

labels = ['Positivity', 'Negativity', 'Personalization', 
          'Superlativeness', 'Eliteness', 'Aesthetic Appeal']
thresholds = {'Positivity': 0.20, 'Negativity': 0.30, 'Personalization': 0.45,
              'Superlativeness': 0.70, 'Eliteness': 0.70, 'Aesthetic Appeal': 0.20}

for i, label in enumerate(labels):
    if probs[i].item() > thresholds[label]:
        print(f"{label}: {probs[i].item():.3f}")
```

## Цитирование

Если вы используете этот репозиторий в научных целях, пожалуйста, укажите ссылку на оригинальную работу:

Долнаков М. В. Автоматическая идентификация визуальных ресурсов для создания новостных ценностей в новостном контенте социальных сетей. Курсовая работа. М.: НИУ ВШЭ, 2026.

## Лицензия

Данный проект распространяется в учебных целях. Автор не даёт гарантий относительно полноты или точности данных и модели.
