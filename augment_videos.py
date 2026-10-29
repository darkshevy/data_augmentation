#!/usr/bin/env python3
"""
Пакетная аугментация коротких MP4-видео через FFmpeg.

Для каждого входного файла создаётся N вариантов со случайными лёгкими
трансформациями:
  1. Полное удаление исходных метаданных (контейнер, потоки, главы, SEI).
  2. Микро-зум (crop) 101–103 % со случайным смещением окна и
     масштабированием обратно в исходное разрешение.
  3. Сдвиг контраста / гаммы / насыщенности на 1–4 % (вверх или вниз).
  4. Цифровое зерно (film grain) с амплитудой 1–2 % от диапазона яркости.
  5. Ускорение видео и аудио в 1.01–1.03 раза (setpts + atempo,
     высота тона звука сохраняется).

Отражения (hflip/vflip/transpose/rotate) не используются никогда: текст в
кадре должен оставаться читаемым. Это дополнительно проверяется перед
каждым запуском FFmpeg.

Каждый выход перекодируется с нуля, поэтому это новый бинарный файл с
другим SHA-256 (проверяется после рендера). Все применённые параметры и
хеши пишутся в manifest.jsonl — так любой вариант можно воспроизвести, а
разметку исходника связать с его аугментациями.

Пример:
    python augment_videos.py ./input ./output --variants 5 --workers 4 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

# Фильтры, которые меняют ориентацию кадра. Запрещены.
FORBIDDEN_FILTERS = ("hflip", "vflip", "transpose", "rotate", "flip")

# Диапазоны трансформаций (из ТЗ).
ZOOM_RANGE = (1.01, 1.03)
COLOR_SHIFT_RANGE = (0.01, 0.04)
NOISE_OPACITY_RANGE = (0.01, 0.02)
SPEED_RANGE = (1.01, 1.03)



@dataclass
class VideoInfo:
    width: int
    height: int
    fps: str  # дробь вида "30000/1001"
    has_audio: bool


@dataclass
class AugmentParams:
    zoom: float
    crop_x_frac: float  # положение окна кропа внутри доступного поля (0..1)
    crop_y_frac: float
    contrast: float
    gamma: float
    saturation: float
    noise_opacity: float
    speed: float
    noise_seed: int


def probe(path: Path) -> VideoInfo:
    """Читает разрешение, fps и наличие аудио через ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("нет видеопотока")
    fps = video.get("avg_frame_rate") or "0/0"
    if fps in ("0/0", "0"):
        fps = video.get("r_frame_rate") or "30/1"
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def random_shift(rng: random.Random) -> float:
    """Множитель 1 ± (1..4 %), знак случайный."""
    delta = rng.uniform(*COLOR_SHIFT_RANGE)
    return 1.0 + delta if rng.random() < 0.5 else 1.0 - delta


def sample_params(rng: random.Random) -> AugmentParams:
    # Сдвигаем от одного до трёх цветовых параметров; остальные остаются 1.0.
    which = rng.sample(["contrast", "gamma", "saturation"], k=rng.randint(1, 3))
    color = {name: (random_shift(rng) if name in which else 1.0)
             for name in ("contrast", "gamma", "saturation")}
    return AugmentParams(
        zoom=rng.uniform(*ZOOM_RANGE),
        crop_x_frac=rng.random(),
        crop_y_frac=rng.random(),
        noise_opacity=rng.uniform(*NOISE_OPACITY_RANGE),
        speed=rng.uniform(*SPEED_RANGE),
        noise_seed=rng.randint(0, 2**31 - 1),
        **color,
    )


def even(n: float) -> int:
    """Ближайшее чётное целое (yuv420p требует чётных размеров)."""
    return max(2, int(round(n / 2)) * 2)


def build_video_filter(info: VideoInfo, p: AugmentParams) -> str:
    out_w, out_h = even(info.width), even(info.height)

    # 2. Микро-зум: вырезаем окно 1/zoom от кадра и растягиваем обратно.
    crop_w = even(info.width / p.zoom)
    crop_h = even(info.height / p.zoom)
    x = int((info.width - crop_w) * p.crop_x_frac)
    y = int((info.height - crop_h) * p.crop_y_frac)

    # 5. Скорость: setpts ускоряет, fps возвращает исходную частоту кадров.
    fps = Fraction(info.fps)
    fps_str = f"{fps.numerator}/{fps.denominator}" if fps > 0 else "30"

    luma_noise = max(1, round(p.noise_opacity * 255))
    chroma_noise = max(1, luma_noise // 2)

    chain = (
        f"[0:v]crop={crop_w}:{crop_h}:{x}:{y},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1,"
        # 3. Цветовая коррекция.
        f"eq=contrast={p.contrast:.4f}:gamma={p.gamma:.4f}:saturation={p.saturation:.4f},"
        f"format=yuv420p,"
        # 4. Зерно с амплитудой 1–2 % от диапазона 0–255 (яркость),
        #    вдвое слабее по цвету; temporal + uniform, как плёночное зерно.
        f"noise=c0s={luma_noise}:c0f=t+u:c1s={chroma_noise}:c1f=t+u:"
        f"c2s={chroma_noise}:c2f=t+u:all_seed={p.noise_seed},"
        f"setpts=PTS/{p.speed:.4f},fps={fps_str}[vout]"
    )
    return chain


def build_audio_filter(p: AugmentParams) -> str:
    # atempo меняет темп без изменения высоты тона.
    return f"[0:a:0]atempo={p.speed:.4f},asetpts=PTS-STARTPTS[aout]"


def assert_no_flip(filtergraph: str) -> None:
    lowered = filtergraph.lower()
    for name in FORBIDDEN_FILTERS:
        if name in lowered:
            raise RuntimeError(f"запрещённый фильтр '{name}' в графе: {filtergraph}")


def build_command(src: Path, dst: Path, info: VideoInfo, p: AugmentParams,
                  crf: int, preset: str) -> list[str]:
    graph = build_video_filter(info, p)
    if info.has_audio:
        graph += ";" + build_audio_filter(p)
    assert_no_flip(graph)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-filter_complex", graph,
        "-map", "[vout]",
    ]
    if info.has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k", "-flags:a", "+bitexact"]

    cmd += [
        # 1. Удаление метаданных: глобальных, потоковых, глав.
        "-map_metadata", "-1",
        "-map_metadata:s:v", "-1",
        "-map_metadata:s:a", "-1",
        "-map_chapters", "-1",
        # Не писать тег encoder=Lavf... в контейнер и потоки.
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        # Видео перекодируется полностью.
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # Убрать SEI-блоки (в т.ч. строку с настройками x264).
        "-bsf:v", "filter_units=remove_types=6",
        "-movflags", "+faststart",
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


def process_one(src: Path, dst: Path, params: AugmentParams, crf: int, preset: str) -> dict:
    info = probe(src)
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    cmd = build_command(src, tmp, info, params, crf, preset)
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
        "params": asdict(params),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Аугментация папки MP4 через FFmpeg (без отражений).")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--variants", type=int, default=3, help="вариантов на файл (по умолчанию 3)")
    ap.add_argument("--workers", type=int, default=2, help="параллельных процессов ffmpeg")
    ap.add_argument("--seed", type=int, default=None, help="seed для воспроизводимости")
    ap.add_argument("--crf", type=int, default=18, help="качество x264 (меньше = лучше)")
    ap.add_argument("--preset", default="medium", help="пресет x264")
    ap.add_argument("--recursive", action="store_true", help="искать MP4 в подпапках")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"Не найден {tool} в PATH", file=sys.stderr)
            return 1

    pattern = "**/*" if args.recursive else "*"
    sources = sorted(p for p in args.input_dir.glob(pattern)
                     if p.is_file() and p.suffix.lower() == ".mp4")
    if not sources:
        print(f"В {args.input_dir} нет MP4-файлов", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Параметры генерируются заранее в одном потоке: при фиксированном seed
    # результат не зависит от порядка выполнения задач.
    jobs = []
    for src in sources:
        rel = src.relative_to(args.input_dir)
        out_dir = args.output_dir / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.variants + 1):
            dst = out_dir / f"{src.stem}_aug{i:02d}.mp4"
            jobs.append((src, dst, sample_params(rng)))

    manifest_path = args.output_dir / "manifest.jsonl"
    lock = threading.Lock()
    ok = failed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_one, s, d, p, args.crf, args.preset): (s, d)
                   for s, d, p in jobs}
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
            print(f"[OK] {dst.name}  zoom={p['zoom']:.3f} speed={p['speed']:.3f} "
                  f"c={p['contrast']:.3f} g={p['gamma']:.3f} s={p['saturation']:.3f} "
                  f"noise={p['noise_opacity']:.3f}")

    print(f"\nГотово: {ok} успешно, {failed} с ошибками. Манифест: {manifest_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
