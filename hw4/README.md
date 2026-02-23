# HW4: Boids

Реализация симуляции `boids` с оптимизацией:

- локальные взаимодействия (`alignment`, `cohesion`, `separation`);
- взаимодействие со стенками (`walls`) и шум (`noise`);
- препятствия двух классов:
  - класс `0` — отталкивающие круги;
  - класс `1` — притягивающие круги;
- JIT-компиляция и распараллеливание через `numba`;
- визуализация в `vispy`;
- запись видео в MP4 через `ffmpeg-python`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Также требуется установить ffmpeg и добавить его в PATH.  
Например, на MacOS можно через Brew:
```bash
brew install ffmpeg
```

Для macOS рекомендуется backend `pyqt6`.
Для Windows рекомендуется backend `pyglet`.

## Запуск интерактивной симуляции

```bash
python main.py --n 1000 --fps 60 --backend pyqt6
```

## Запись видео

### N = 1000, 1800 кадров (30 секунд при 60 fps)

```bash
python main.py \
  --n 1000 \
  --fps 60 \
  --steps 1800 \
  --record video_N1000.mp4 \
  --backend pyqt6
```

### N = 5000, 3600 кадров (60 секунд при 60 fps)

```bash
python main.py \
  --n 5000 \
  --fps 60 \
  --steps 3600 \
  --record video_N5000.mp4 \
  --backend pyqt6
```

## Настройка параметров модели

Все коэффициенты поведения можно менять в файле:

- `boids_config.py` (`class BoidsParams`)

Основные параметры:

- `alignment_weight` — сила выравнивания скорости по локальным соседям.
- `cohesion_weight` — сила притяжения к локальному центру группы.
- `separation_weight` — сила отталкивания от слишком близких соседей.
- `wall_weight` — сила возврата от границ области.
- `noise_weight` — сила случайной компоненты (хаотичность траекторий).
- `repulsive_obstacle_weight` — сила отталкивания от препятствий класса `0` (красные).
- `attractive_obstacle_weight` — сила притяжения к препятствиям класса `1` (зеленые).
- `max_force` — ограничение максимального ускорения/маневра за шаг.
- `max_speed` — ограничение максимальной скорости агента.
- `perception_radius` — радиус видимости соседей для `alignment/cohesion`.
- `separation_radius` — ближний радиус для `separation` (обычно меньше `perception_radius`).
- `wall_margin` — толщина зоны у стен, где начинает действовать `wall_weight`.
- `obstacle_padding` — дополнительная зона влияния препятствий (`radius + padding`).
- `dt` — шаг интегрирования по времени (при `1/60` модель обновляется 60 раз в секунду).

Препятствия по умолчанию настраиваются в:

- `boids_config.py` (`default_obstacles(width, height)`)

Там можно изменить:

- число кругов,
- координаты центров,
- радиусы,
- классы препятствий (`0` — отталкивает, `1` — притягивает).

## Структура проекта

- `main.py` — точка входа, парсинг CLI, запуск визуализации.
- `boids_config.py` — конфигурация параметров и генерация препятствий.
- `boids_core.py` — вычислительное ядро, реализация самой симуляция Boids.
- `boids_runtime.py` - рендер моделирования.
- `ffmpeg_recorder.py` — запись MP4.

## Записи примеров

1. 1000 агентов, 3600 кадров, 60 кадров в секунду, остальные настройки по умолчанию:   
  https://drive.google.com/file/d/1mb_sYxTcbp6RKWQ_KrFOWsCWSx0Mim2k/view?usp=sharing
2. 5000 агентов, 3600 кадров, 60 кадров в секунду, остальные настройки по умолчанию:  
  https://drive.google.com/file/d/1j7rc59iekqX4zAOQNFfVx7ILJODT-ZV5/view?usp=sharing