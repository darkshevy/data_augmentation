#!/usr/bin/env python3
"""
Пакетная аугментация коротких MP4-видео через FFmpeg.

Для каждого входного файла создаётся N вариантов со случайными лёгкими
трансформациями, повышающими устойчивость к пространственным, цветовым,
временным и акустическим искажениям:

  Контейнер
  1. Полное удаление исходных метаданных (контейнер, потоки, главы) и
     подстановка случайного профиля мобильного устройства
     (Apple iPhone 14/15/16 Pro + creation_time за последние 48 часов).
  2. Случайный CRF (±1 от базового) и случайный размер GOP (48..120).

  Время
  3. Случайная обрезка 0.2–0.6 с в начале и/или в конце (для роликов
     длиннее 4.5 с) через -ss/-t.
  4. Ускорение видео и аудио в 1.015–1.045 раза.

  Пространство
  5. Плавный синусоидальный зум 1.04–1.09 (scale с eval=frame).
  6. Микро-поворот на 0.5–1.3° в любую сторону с последующей обрезкой до
     исходного размера (зум гарантированно перекрывает чёрные углы).
  7. Необязательный режим фоновой рамки: основное видео уменьшается до
     91–95 % кадра и накладывается по центру на сильно размытую
     (gblur=sigma=25), отмасштабированную и затемнённую копию потока.

  Цвет и текстура
  8. Сдвиг контраста / гаммы / насыщенности на 4–9 %, сдвиг оттенка на
     ±4°, лёгкий unsharp.
  9. Цифровое зерно (c0s=5..9, c1s/c2s=2..4 — переживает квантование
     H.264) и мягкая виньетка со случайным углом.

  Звук
 10. Сдвиг высоты тона на ±1.8..3.5 % (asetrate+aresample) с компенсацией
     темпа через atempo, 2–3 случайных полосовых equalizer, подмешанный
     тихий розовый шум и случайный битрейт AAC.

  Текст
 11. Необязательный слой drawtext со случайной строкой из --captions-file.

Зеркалирования (hflip/vflip/transpose) не используются никогда: текст в
кадре должен оставаться читаемым. Это проверяется перед каждым запуском
FFmpeg. Микро-поворот (rotate) разрешён — он не зеркалит кадр.

Каждый выход перекодируется с нуля, поэтому это новый бинарный файл с
другим SHA-256 (проверяется после рендера). Все применённые параметры и
хеши пишутся в manifest.jsonl — так любой вариант можно воспроизвести, а
разметку исходника связать с его аугментациями.

Пример:
    python augment_videos.py ./input ./output --variants 5 --workers 4 --seed 42
    python augment_videos.py ./input ./output --frame-style on --captions-file texts.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

# Фильтры, которые зеркалят кадр. Запрещены всегда.
# rotate сюда НЕ входит: микро-поворот не делает текст нечитаемым.
FORBIDDEN_FILTERS = ("hflip", "vflip", "transpose")

# Имя фильтра стоит либо в начале цепочки, либо сразу после ',', ';' или ']'.
# Экранированные разделители (они встречаются только внутри текста drawtext)
# не считаются позицией имени фильтра.
_FORBIDDEN_RE = re.compile(
    r"(?:^|(?<!\\)[,;\]])\s*(" + "|".join(FORBIDDEN_FILTERS) + r")\b",
    re.IGNORECASE,
)

# --- Диапазоны трансформаций -------------------------------------------------

ZOOM_RANGE = (1.04, 1.09)          # амплитуда синусоидального зума
ZOOM_PERIOD_RANGE = (3.0, 9.0)     # период колебания зума, секунды
ROTATE_DEG_RANGE = (0.5, 1.3)      # |угол| микро-поворота, градусы
ZOOM_SAFETY = 1.004                # запас зума над геометрическим минимумом

FRAME_SCALE_RANGE = (0.91, 0.95)   # доля кадра под основное видео в режиме рамки
FRAME_BG_ZOOM_RANGE = (1.05, 1.15) # масштаб фоновой копии до центрального кропа
FRAME_BLUR_SIGMA = 25.0
FRAME_BG_BRIGHTNESS_RANGE = (-0.16, -0.04)
FRAME_BG_SATURATION_RANGE = (0.75, 1.15)
FRAME_STYLE_PROBABILITY = 0.5

COLOR_SHIFT_RANGE = (0.04, 0.09)   # контраст / гамма / насыщенность
HUE_DEG_RANGE = (-4.0, 4.0)        # сдвиг оттенка, градусы
UNSHARP_PARAMS = "5:5:0.35:5:5:0.0"

NOISE_LUMA_RANGE = (5, 9)          # c0s — зерно переживает квантование H.264
NOISE_CHROMA_RANGE = (2, 4)        # c1s / c2s
VIGNETTE_ANGLE_RANGE = (math.pi / 14, math.pi / 8)

SPEED_RANGE = (1.015, 1.045)
TRIM_MIN_DURATION = 4.5            # короче — не режем
TRIM_RANGE = (0.2, 0.6)            # секунд с каждого выбранного конца

PITCH_SHIFT_RANGE = (0.018, 0.035) # ±1.8..3.5 % по высоте тона
EQ_BANDS = ((200.0, 400.0), (1000.0, 2500.0), (4000.0, 7000.0))
EQ_GAIN_RANGE = (2.0, 4.0)         # ±dB
EQ_Q_RANGE = (0.7, 1.6)
PINK_NOISE_AMPLITUDE = 0.0012
AUDIO_BITRATES = ("128k", "160k", "192k")
DEFAULT_SAMPLE_RATE = 48000

GOP_RANGE = (48, 120)
CRF_JITTER = (-1, 0, 1)
CRF_MIN = 14

# Профили метаданных мобильных устройств (пишутся вместо вычищенных).
DEVICE_PRESETS = (
    {"make": "Apple", "model": "iPhone 14 Pro", "software": "17.6.1"},
    {"make": "Apple", "model": "iPhone 15 Pro", "software": "18.1.1"},
    {"make": "Apple", "model": "iPhone 16 Pro", "software": "18.3.2"},
)
METADATA_MAX_AGE_HOURS = 48

# Шрифты для drawtext (берётся первый существующий, либо --font).
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)

CAPTION_FONT_SCALE_RANGE = (0.038, 0.062)  # доля высоты кадра
CAPTION_MARGIN_RANGE = (0.04, 0.11)        # доля высоты кадра
CAPTION_BORDER_RANGE = (1, 3)
CAPTION_MAX_CHARS = 120
# drawtext не переносит текст сам. Ширину строки считаем из ширины кадра:
# 0.86 — безопасная зона по горизонтали, 0.58 — средняя ширина глифа в
# долях кегля.
CAPTION_SAFE_WIDTH = 0.86
CAPTION_GLYPH_RATIO = 0.58
CAPTION_MIN_CHARS_PER_LINE = 12

DEFAULT_PRESET = "medium"
# h264_nvenc на строгих сборках FFmpeg не принимает пресеты libx264.
NVENC_DEFAULT_PRESET = "p4"


# --- Структуры данных --------------------------------------------------------


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: str              # дробь вида "30000/1001"
    duration: float       # секунды
    has_audio: bool
    sample_rate: int


@dataclass
class EqBand:
    freq: float
    gain: float
    q: float


@dataclass
class MetadataProfile:
    make: str
    model: str
    software: str
    creation_time: str    # ISO 8601, UTC


@dataclass
class CaptionParams:
    text: str             # уже разбитый на строки (textwrap)
    fontsize: int
    max_chars_per_line: int
    position: str         # "top" | "bottom"
    margin: int           # пикселей от края
    borderw: int


@dataclass
class AugmentParams:
    # Геометрия
    zoom_lo: float
    zoom_hi: float
    zoom_period: float
    zoom_phase: float
    rotate_deg: float
    # Режим фоновой рамки
    frame_style: bool
    frame_scale: float
    frame_bg_zoom: float
    frame_blur_sigma: float
    frame_bg_brightness: float
    frame_bg_saturation: float
    # Цвет и резкость
    contrast: float
    gamma: float
    saturation: float
    hue_deg: float
    unsharp: str
    # Зерно и виньетка
    noise_luma: int
    noise_chroma: int
    noise_seed: int
    vignette_angle: float
    # Время
    speed: float
    trim_start: float
    trim_end: float
    out_duration: Optional[float]   # None — читаем файл целиком
    # Звук
    pitch: float
    atempo: float
    eq_bands: List[EqBand]
    pink_amplitude: float
    audio_bitrate: str
    # Контейнер / кодек
    crf: int
    gop: int
    metadata: MetadataProfile
    # Текстовый слой
    caption: Optional[CaptionParams] = None


@dataclass
class RenderOptions:
    preset: str = "medium"
    nvenc: bool = False
    font: Optional[str] = None
    extra: dict = field(default_factory=dict)


# --- ffprobe -----------------------------------------------------------------


def probe(path: Path) -> VideoInfo:
    """Читает разрешение, fps, длительность, наличие и частоту аудио."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,sample_rate,duration"
        ":format=duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("нет видеопотока")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = video.get("avg_frame_rate") or "0/0"
    if fps in ("0/0", "0", ""):
        fps = video.get("r_frame_rate") or "30/1"
        # В MP4 без avg_frame_rate r_frame_rate часто равен timebase (90000/1):
        # это не частота кадров, а тики. Берём безопасные 30 fps.
        if to_float_fps(fps) > 120.0:
            fps = "30/1"
    if to_float_fps(fps) <= 0.0:
        fps = "30/1"

    duration = _as_float(data.get("format", {}).get("duration"))
    if duration <= 0.0:
        duration = _as_float(video.get("duration"))

    sample_rate = DEFAULT_SAMPLE_RATE
    if audio is not None:
        try:
            sample_rate = int(audio.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            sample_rate = DEFAULT_SAMPLE_RATE
        if sample_rate <= 0:
            sample_rate = DEFAULT_SAMPLE_RATE

    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        has_audio=audio is not None,
        sample_rate=sample_rate,
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_float_fps(value: str) -> float:
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


# --- Генерация параметров ----------------------------------------------------


def random_shift(rng: random.Random) -> float:
    """Множитель 1 ± (4..9 %), знак случайный."""
    delta = rng.uniform(*COLOR_SHIFT_RANGE)
    return 1.0 + delta if rng.random() < 0.5 else 1.0 - delta


def min_zoom_for_rotation(width: int, height: int, deg: float) -> float:
    """
    Минимальный зум, при котором центральный кроп исходного размера
    целиком помещается внутрь повёрнутого кадра (без чёрных углов).

    Прямоугольник a*W x a*H вписан в повёрнутый на угол t прямоугольник WxH,
    если a*(W*cos+H*sin) <= W и a*(W*sin+H*cos) <= H.
    """
    rad = math.radians(abs(deg))
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    return max(
        (width * cos_t + height * sin_t) / width,
        (width * sin_t + height * cos_t) / height,
    )


def sample_metadata(rng: random.Random) -> MetadataProfile:
    preset = rng.choice(DEVICE_PRESETS)
    age = timedelta(seconds=rng.uniform(0, METADATA_MAX_AGE_HOURS * 3600))
    stamp = datetime.now(timezone.utc) - age
    return MetadataProfile(
        make=preset["make"],
        model=preset["model"],
        software=preset["software"],
        creation_time=stamp.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def sample_caption(rng: random.Random, info: VideoInfo,
                   captions: List[str]) -> Optional[CaptionParams]:
    if not captions:
        return None
    width, height = even(info.width), even(info.height)
    text = rng.choice(captions)[:CAPTION_MAX_CHARS]
    fontsize = max(10, int(round(height * rng.uniform(*CAPTION_FONT_SCALE_RANGE))))
    # Длинная строка иначе уезжает за границы кадра (особенно вертикального),
    # поэтому раскладываем её на несколько строк заранее.
    max_chars_per_line = max(
        CAPTION_MIN_CHARS_PER_LINE,
        int((width * CAPTION_SAFE_WIDTH) / (fontsize * CAPTION_GLYPH_RATIO)),
    )
    return CaptionParams(
        text=textwrap.fill(text, width=max_chars_per_line),
        fontsize=fontsize,
        max_chars_per_line=max_chars_per_line,
        position=rng.choice(("top", "bottom")),
        margin=max(4, int(round(height * rng.uniform(*CAPTION_MARGIN_RANGE)))),
        borderw=rng.randint(*CAPTION_BORDER_RANGE),
    )


def sample_eq_bands(rng: random.Random) -> List[EqBand]:
    bands = rng.sample(EQ_BANDS, k=rng.randint(2, 3))
    out = []
    for lo, hi in bands:
        gain = rng.uniform(*EQ_GAIN_RANGE)
        out.append(EqBand(
            freq=rng.uniform(lo, hi),
            gain=gain if rng.random() < 0.5 else -gain,
            q=rng.uniform(*EQ_Q_RANGE),
        ))
    return out


def sample_params(rng: random.Random, info: VideoInfo, *, base_crf: int,
                  frame_style_mode: str, captions: List[str]) -> AugmentParams:
    # Сдвигаем от одного до трёх цветовых параметров; остальные остаются 1.0.
    which = rng.sample(["contrast", "gamma", "saturation"], k=rng.randint(1, 3))
    color = {name: (random_shift(rng) if name in which else 1.0)
             for name in ("contrast", "gamma", "saturation")}

    # Поворот: знак случайный, модуль из [0.5, 1.3] градуса.
    rotate_deg = rng.uniform(*ROTATE_DEG_RANGE)
    if rng.random() < 0.5:
        rotate_deg = -rotate_deg

    # Зум обязан перекрывать поворот, иначе в углах появится чёрный фон.
    need = min_zoom_for_rotation(info.width, info.height, rotate_deg) * ZOOM_SAFETY
    zoom_lo = max(ZOOM_RANGE[0], need)
    zoom_hi = max(ZOOM_RANGE[1], zoom_lo + 0.005)

    # Временная обрезка — только если ролик заметно длиннее порога.
    trim_start = trim_end = 0.0
    out_duration: Optional[float] = None
    if info.duration > TRIM_MIN_DURATION:
        mode = rng.choice(("start", "end", "both"))
        if mode in ("start", "both"):
            trim_start = rng.uniform(*TRIM_RANGE)
        if mode in ("end", "both"):
            trim_end = rng.uniform(*TRIM_RANGE)
        out_duration = max(1.0, info.duration - trim_start - trim_end)

    if frame_style_mode == "on":
        frame_style = True
    elif frame_style_mode == "off":
        frame_style = False
    else:
        frame_style = rng.random() < FRAME_STYLE_PROBABILITY

    speed = rng.uniform(*SPEED_RANGE)
    pitch_delta = rng.uniform(*PITCH_SHIFT_RANGE)
    pitch = 1.0 + pitch_delta if rng.random() < 0.5 else 1.0 - pitch_delta

    return AugmentParams(
        zoom_lo=zoom_lo,
        zoom_hi=zoom_hi,
        zoom_period=rng.uniform(*ZOOM_PERIOD_RANGE),
        zoom_phase=rng.uniform(0.0, 2 * math.pi),
        rotate_deg=rotate_deg,
        frame_style=frame_style,
        frame_scale=rng.uniform(*FRAME_SCALE_RANGE),
        frame_bg_zoom=rng.uniform(*FRAME_BG_ZOOM_RANGE),
        frame_blur_sigma=FRAME_BLUR_SIGMA,
        frame_bg_brightness=rng.uniform(*FRAME_BG_BRIGHTNESS_RANGE),
        frame_bg_saturation=rng.uniform(*FRAME_BG_SATURATION_RANGE),
        hue_deg=rng.uniform(*HUE_DEG_RANGE),
        unsharp=UNSHARP_PARAMS,
        noise_luma=rng.randint(*NOISE_LUMA_RANGE),
        noise_chroma=rng.randint(*NOISE_CHROMA_RANGE),
        noise_seed=rng.randint(0, 2**31 - 1),
        vignette_angle=rng.uniform(*VIGNETTE_ANGLE_RANGE),
        speed=speed,
        trim_start=trim_start,
        trim_end=trim_end,
        out_duration=out_duration,
        pitch=pitch,
        atempo=speed / pitch,
        eq_bands=sample_eq_bands(rng),
        pink_amplitude=PINK_NOISE_AMPLITUDE,
        audio_bitrate=rng.choice(AUDIO_BITRATES),
        crf=max(CRF_MIN, base_crf + rng.choice(CRF_JITTER)),
        gop=rng.randint(*GOP_RANGE),
        metadata=sample_metadata(rng),
        caption=sample_caption(rng, info, captions),
        **color,
    )


# --- Построение filter_complex ----------------------------------------------


def even(n: float) -> int:
    """Ближайшее чётное целое (yuv420p требует чётных размеров)."""
    return max(2, int(round(n / 2)) * 2)


def escape_filter_text(text: str) -> str:
    """
    Экранирует произвольную строку для значения опции фильтра в
    -filter_complex.

    Строка проходит два разбора: сначала граф фильтров (разделители
    ',', ';', '[', ']', кавычка и обратный слеш), затем парсер опций
    конкретного фильтра (разделитель ':', кавычка и обратный слеш).
    Поэтому символы второго уровня нуждаются в двойном экранировании.
    """
    out = text.replace("\\", "\\\\\\\\")   # \  -> \\\\
    out = out.replace("'", "\\\\\\'")      # '  -> \\\'
    out = out.replace(":", "\\\\:")        # :  -> \\:
    out = out.replace("%", "\\\\%")        # %  -> \\%
    for ch in (",", ";", "[", "]"):
        out = out.replace(ch, "\\" + ch)   # ,  -> \,
    return out


def build_video_filter(info: VideoInfo, p: AugmentParams,
                       font: Optional[str]) -> str:
    out_w, out_h = even(info.width), even(info.height)

    fps = Fraction(info.fps)
    fps_str = f"{fps.numerator}/{fps.denominator}" if fps > 0 else "30"

    # Плавный зум: z(t) = lo + (hi-lo) * (1 + sin(2*PI*t/period + phase)) / 2.
    # Выражение намеренно построено без запятых (sin/trunc — одноаргументные),
    # но дополнительно закавычено: кавычки защищают ',' на уровне графа.
    span = p.zoom_hi - p.zoom_lo
    zexpr = (f"({p.zoom_lo:.5f}+{span:.5f}"
             f"*(1+sin(2*PI*t/{p.zoom_period:.4f}+{p.zoom_phase:.4f}))/2)")

    # Микро-поворот: кадр остаётся того же размера, чёрные углы срежет кроп.
    rotate = (f"rotate=a={math.radians(p.rotate_deg):.6f}"
              f":ow=iw:oh=ih:c=black@1.0")
    # Динамический масштаб + центральный кроп до исходного размера.
    # crop пересчитывает x/y на каждом кадре, поэтому окно всегда по центру.
    zoom = (f"scale=w='trunc(iw*{zexpr}/2)*2':h='trunc(ih*{zexpr}/2)*2'"
            f":eval=frame:flags=bicubic")
    crop = (f"crop={out_w}:{out_h}:x='(in_w-out_w)/2':y='(in_h-out_h)/2'"
            f":exact=1")

    color = (f"eq=contrast={p.contrast:.4f}:gamma={p.gamma:.4f}"
             f":saturation={p.saturation:.4f},"
             f"hue=h={p.hue_deg:.3f},"
             f"unsharp={p.unsharp}")

    grain = (f"noise=c0s={p.noise_luma}:c0f=t+u:c1s={p.noise_chroma}:c1f=t+u:"
             f"c2s={p.noise_chroma}:c2f=t+u:all_seed={p.noise_seed}")
    vignette = f"vignette=angle={p.vignette_angle:.5f}:mode=forward:eval=init"

    tail = ["format=yuv420p", grain, vignette]
    caption_filter = build_caption_filter(p.caption, font)
    if caption_filter:
        tail.append(caption_filter)
    tail.append(f"setpts=PTS/{p.speed:.4f}")
    tail.append(f"fps={fps_str}")
    tail_chain = ",".join(tail)

    main = f"{rotate},{zoom},{crop},setsar=1,{color}"

    if not p.frame_style:
        return f"[0:v]{main},{tail_chain}[vout]"

    # Режим фоновой рамки: основное видео уменьшается и ложится по центру
    # на размытую, отмасштабированную и затемнённую копию того же потока.
    fg_w, fg_h = even(out_w * p.frame_scale), even(out_h * p.frame_scale)
    bg_w, bg_h = even(out_w * p.frame_bg_zoom), even(out_h * p.frame_bg_zoom)
    return ";".join([
        "[0:v]split=2[bgsrc][fgsrc]",
        f"[fgsrc]{main},scale={fg_w}:{fg_h}:flags=lanczos[fg]",
        (f"[bgsrc]scale={bg_w}:{bg_h}:flags=bicubic,crop={out_w}:{out_h},"
         f"gblur=sigma={p.frame_blur_sigma:g}:steps=2,"
         f"eq=brightness={p.frame_bg_brightness:.4f}"
         f":saturation={p.frame_bg_saturation:.4f},setsar=1[bg]"),
        (f"[bg][fg]overlay=x='(W-w)/2':y='(H-h)/2':format=auto:shortest=0,"
         f"{tail_chain}[vout]"),
    ])


def escape_font_path(path: str) -> str:
    """
    Готовит путь к шрифту для опции fontfile в -filter_complex.

    Windows-путь вида C:/Windows/Fonts/arialbd.ttf парсер опций фильтра
    обрезает по двоеточию после буквы диска, причём FFmpeg на этом не
    падает, а молча подставляет другой шрифт. Кавычки не помогают —
    внутри них экранирование не разбирается, — поэтому путь
    подставляется без кавычек, а ':' и кавычка экранируются.
    """
    normalized = path.replace("\\", "/")
    return normalized.replace("'", "\\\\\\'").replace(":", "\\\\:")


def build_caption_filter(caption: Optional[CaptionParams],
                         font: Optional[str]) -> str:
    if caption is None:
        return ""
    if not font:
        raise RuntimeError("для --captions-file нужен шрифт (--font)")
    if caption.position == "top":
        y_expr = f"{caption.margin}"
    else:
        y_expr = f"h-text_h-{caption.margin}"
    return (
        f"drawtext=fontfile={escape_font_path(font)}"
        f":text={escape_filter_text(caption.text)}"
        f":expansion=none"
        f":fontsize={caption.fontsize}"
        f":fontcolor=white@0.96"
        f":borderw={caption.borderw}:bordercolor=black@0.85"
        f":x='(w-text_w)/2':y='{y_expr}'"
    )


def build_audio_filter(info: VideoInfo, p: AugmentParams) -> str:
    sr = info.sample_rate
    # Сдвиг высоты тона через asetrate (меняет и темп), затем компенсация
    # темпа: итоговая скорость звука совпадает со скоростью видео.
    chain = [
        f"asetrate={sr}*{p.pitch:.5f}",
        f"aresample={sr}",
        f"atempo={p.atempo:.6f}",
    ]
    for band in p.eq_bands:
        chain.append(f"equalizer=f={band.freq:.1f}:width_type=q"
                     f":width={band.q:.3f}:g={band.gain:.2f}")
    chain.append("asetpts=PTS-STARTPTS")
    voice = f"[0:a:0]{','.join(chain)}[a0]"

    # Тихий розовый шум как «шум записи». normalize=0 — чтобы amix
    # не ронял громкость голоса вдвое.
    noise = (f"anoisesrc=color=pink:amplitude={p.pink_amplitude:g}"
             f":sample_rate={sr}[apink]")
    mix = ("[a0][apink]amix=inputs=2:duration=first:dropout_transition=0"
           ":normalize=0[aout]")
    return ";".join([voice, noise, mix])


def assert_no_flip(filtergraph: str) -> None:
    """Проверяет, что в графе нет зеркалящих фильтров (rotate разрешён)."""
    match = _FORBIDDEN_RE.search(filtergraph)
    if match:
        raise RuntimeError(
            f"запрещённый фильтр '{match.group(1)}' в графе: {filtergraph}")


# --- Сборка команды ----------------------------------------------------------


def build_command(src: Path, dst: Path, info: VideoInfo, p: AugmentParams,
                  opts: RenderOptions) -> List[str]:
    graph = build_video_filter(info, p, opts.font)
    if info.has_audio:
        graph += ";" + build_audio_filter(info, p)
    assert_no_flip(graph)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    # Временная обрезка задаётся на входе: -ss сдвигает начало, -t
    # ограничивает читаемый кусок, поэтому t в фильтрах снова идёт с нуля.
    if p.trim_start > 0:
        cmd += ["-ss", f"{p.trim_start:.3f}"]
    if p.out_duration is not None:
        cmd += ["-t", f"{p.out_duration:.3f}"]
    cmd += ["-i", str(src), "-filter_complex", graph, "-map", "[vout]"]

    if info.has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", p.audio_bitrate,
                "-ar", str(info.sample_rate)]

    cmd += [
        # 1. Удаление метаданных: глобальных, потоковых, глав.
        "-map_metadata", "-1",
        "-map_metadata:s:v", "-1",
        "-map_metadata:s:a", "-1",
        "-map_chapters", "-1",
        # ...и подстановка профиля мобильного устройства вместо них.
        "-metadata", f"com.apple.quicktime.make={p.metadata.make}",
        "-metadata", f"com.apple.quicktime.model={p.metadata.model}",
        "-metadata", f"com.apple.quicktime.software={p.metadata.software}",
        "-metadata", f"creation_time={p.metadata.creation_time}",
    ]

    if opts.nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", opts.preset,
                "-rc", "vbr", "-cq", str(p.crf), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", opts.preset, "-crf", str(p.crf)]

    cmd += [
        "-g", str(p.gop),
        "-pix_fmt", "yuv420p",
        # use_metadata_tags обязателен: без него mp4-мультиплексор
        # выбрасывает нестандартные ключи com.apple.quicktime.*
        "-movflags", "+faststart+use_metadata_tags",
        "-sn", "-dn",
        str(dst),
    ]
    return cmd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_one(src: Path, dst: Path, info: VideoInfo, params: AugmentParams,
                opts: RenderOptions) -> dict:
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    cmd = build_command(src, tmp, info, params, opts)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "ffmpeg завершился с ошибкой")

    src_hash, dst_hash = sha256(src), sha256(tmp)
    if src_hash == dst_hash:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("хеш выхода совпал с исходником")
    tmp.replace(dst)

    return {
        "source": str(src),
        "output": str(dst),
        "source_sha256": src_hash,
        "output_sha256": dst_hash,
        "has_audio": info.has_audio,
        "source_info": asdict(info),
        "encoder": "h264_nvenc" if opts.nvenc else "libx264",
        "preset": opts.preset,
        "params": asdict(params),
        "ffmpeg_cmd": cmd,
    }


# --- CLI ---------------------------------------------------------------------


def resolve_font(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit(f"Шрифт не найден: {explicit}")
        return explicit
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def load_captions(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.is_file():
        raise SystemExit(f"Файл с текстами не найден: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    captions = [line for line in lines if line]
    if not captions:
        raise SystemExit(f"В {path} нет непустых строк")
    return captions


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Аугментация папки MP4 через FFmpeg (без зеркалирования).")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--variants", type=int, default=3, help="вариантов на файл (по умолчанию 3)")
    ap.add_argument("--workers", type=int, default=2, help="параллельных процессов ffmpeg")
    ap.add_argument("--seed", type=int, default=None, help="seed для воспроизводимости")
    ap.add_argument("--crf", type=int, default=18,
                    help="базовое качество x264 (к нему добавляется ±1)")
    ap.add_argument("--preset", default=DEFAULT_PRESET,
                    help=f"пресет кодировщика (по умолчанию {DEFAULT_PRESET}, "
                         f"с --nvenc — {NVENC_DEFAULT_PRESET})")
    ap.add_argument("--nvenc", action="store_true",
                    help="кодировать через h264_nvenc (GPU) вместо libx264")
    ap.add_argument("--frame-style", choices=("off", "on", "random"), default="random",
                    help="фоновая рамка с размытой копией кадра (random = 50 %%)")
    ap.add_argument("--captions-file", type=Path, default=None,
                    help="текстовый файл: случайная строка накладывается через drawtext")
    ap.add_argument("--font", default=None,
                    help="путь к TTF для drawtext (по умолчанию ищется системный)")
    ap.add_argument("--recursive", action="store_true", help="искать MP4 в подпапках")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"Не найден {tool} в PATH", file=sys.stderr)
            return 1

    captions = load_captions(args.captions_file)
    font = resolve_font(args.font)
    if captions and not font:
        print("Не найден шрифт для drawtext — укажите --font /path/to/font.ttf",
              file=sys.stderr)
        return 1

    pattern = "**/*" if args.recursive else "*"
    sources = sorted(p for p in args.input_dir.glob(pattern)
                     if p.is_file() and p.suffix.lower() == ".mp4")
    if not sources:
        print(f"В {args.input_dir} нет MP4-файлов", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    preset = args.preset
    if args.nvenc and preset == DEFAULT_PRESET:
        # Пресет не задавали явно: подставляем безопасный для h264_nvenc.
        preset = NVENC_DEFAULT_PRESET
        print(f"--nvenc: пресет по умолчанию заменён на {preset}", file=sys.stderr)
    opts = RenderOptions(preset=preset, nvenc=args.nvenc, font=font)

    # Параметры генерируются заранее в одном потоке: при фиксированном seed
    # результат не зависит от порядка выполнения задач. Каждый исходник
    # зондируется ровно один раз — от его размера и длительности зависят
    # границы зума и обрезки.
    jobs = []
    skipped = 0
    for src in sources:
        try:
            info = probe(src)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[SKIP] {src.name}: {exc}", file=sys.stderr)
            continue
        rel = src.relative_to(args.input_dir)
        out_dir = args.output_dir / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.variants + 1):
            dst = out_dir / f"{src.stem}_aug{i:02d}.mp4"
            params = sample_params(rng, info, base_crf=args.crf,
                                   frame_style_mode=args.frame_style,
                                   captions=captions)
            jobs.append((src, dst, info, params))

    if not jobs:
        print("Нечего обрабатывать", file=sys.stderr)
        return 1

    manifest_path = args.output_dir / "manifest.jsonl"
    lock = threading.Lock()
    ok = failed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_one, s, d, i, p, opts): (s, d)
                   for s, d, i, p in jobs}
        for fut in as_completed(futures):
            src, dst = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {src.name} -> {dst.name}: {exc}", file=sys.stderr)
                continue
            ok += 1
            with lock:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
            p = record["params"]
            print(
                f"[OK] {dst.name}  zoom={p['zoom_lo']:.3f}-{p['zoom_hi']:.3f} "
                f"rot={p['rotate_deg']:+.2f}° speed={p['speed']:.3f} "
                f"pitch={p['pitch']:.3f} hue={p['hue_deg']:+.1f}° "
                f"crf={p['crf']} gop={p['gop']} "
                f"frame={'on' if p['frame_style'] else 'off'} "
                f"trim={p['trim_start']:.2f}/{p['trim_end']:.2f} "
                f"cap={'yes' if p['caption'] else 'no'}"
            )

    print(f"\nГотово: {ok} успешно, {failed} с ошибками"
          + (f", {skipped} пропущено" if skipped else "")
          + f". Манифест: {manifest_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
