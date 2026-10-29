# data_augmentation
## augment_videos.py

Пакетная аугментация MP4 через FFmpeg (без отражений — текст в кадре остаётся читаемым).

Требования: Python 3.9+, `ffmpeg` и `ffprobe` в PATH.

```bash
python augment_videos.py ./input ./output --variants 5 --workers 4 --seed 42
```

Каждый вариант: удаление метаданных, микро-зум 101–103 %, сдвиг контраста/гаммы/насыщенности
на 1–4 %, зерно 1–2 %, ускорение видео и аудио в 1.01–1.03 раза. Параметры и SHA-256 каждого
файла пишутся в `output/manifest.jsonl`.
