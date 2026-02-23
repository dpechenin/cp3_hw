# HW4: Boids

Реализация симуляции `boids` с оптимизацией:

- локальные взаимодействия (`alignment`, `cohesion`, `separation`);
- взаимодействие со стенками (`walls`) и шум (`noise`);
- препятствия двух классов:
  - класс `0` — отталкивающие круги;
  - класс `1` — притягивающие круги;
- JIT-компиляция и распараллеливание через `numba` (`@njit`, `prange`);
- визуализация в `vispy` + текст в окне:
  - число агентов,
  - параметры взаимодействий,
  - FPS;
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
  --record video_N1000_1800.mp4 \
  --backend pyqt6
```

### N = 5000, 3600 кадров (60 секунд при 60 fps)

```bash
python main.py \
  --n 5000 \
  --fps 60 \
  --steps 3600 \
  --record video_N5000_3600.mp4 \
  --backend pyqt6
```

## Структура проекта

- `main.py` — точка входа, парсинг CLI, запуск визуализации.
- `boids_config.py` — конфигурация параметров и генерация препятствий.
- `boids_core.py` — вычислительное ядро (`numba`), реализация самой симуляция Boids.
- `boids_runtime.py` - рендер моделирования.
- `ffmpeg_recorder.py` — запись MP4.

## Записи примеров


