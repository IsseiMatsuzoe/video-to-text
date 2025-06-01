#!/usr/bin/env python3
"""
OpenAI Whisper APIを使用した動画・音声転写CLIツール

長時間の動画ファイルを効率的に転写するため、音声圧縮とチャンク分割機能を提供します。
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import math
import urllib.request
import zipfile
import platform

try:
    import openai
    from tqdm import tqdm
    from dotenv import load_dotenv
    
    # .envファイルを自動読み込み
    load_dotenv()
    
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("pip install -r requirements.txt を実行してください")
    sys.exit(1)


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WhisperError(Exception):
    """Whisper関連のエラー"""
    pass


class FFmpegError(Exception):
    """FFmpeg関連のエラー"""
    pass


def download_ffmpeg_portable():
    """ポータブル版ffmpegを自動ダウンロード"""
    ffmpeg_dir = Path("ffmpeg_portable")
    ffmpeg_exe = ffmpeg_dir / "bin" / "ffmpeg.exe"
    ffprobe_exe = ffmpeg_dir / "bin" / "ffprobe.exe"
    
    if ffmpeg_exe.exists() and ffprobe_exe.exists():
        logger.info("ポータブル版ffmpegが既に存在します")
        return ffmpeg_dir / "bin"
    
    logger.info("ポータブル版ffmpegをダウンロード中...")
    
    # Windows 64bit用のffmpegダウンロードURL
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        # ダウンロード
        zip_path = "ffmpeg_temp.zip"
        urllib.request.urlretrieve(url, zip_path)
        logger.info("ダウンロード完了")
        
        # 展開
        logger.info("展開中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_extract")
        
        # 必要なファイルのみをコピー
        import shutil
        extracted_dir = Path("temp_extract")
        ffmpeg_folder = next(extracted_dir.glob("ffmpeg-*"))
        
        ffmpeg_dir.mkdir(exist_ok=True)
        shutil.copytree(ffmpeg_folder / "bin", ffmpeg_dir / "bin", dirs_exist_ok=True)
        
        # 一時ファイルを削除
        os.remove(zip_path)
        shutil.rmtree("temp_extract")
        
        logger.info(f"ポータブル版ffmpegの準備完了: {ffmpeg_dir / 'bin'}")
        return ffmpeg_dir / "bin"
        
    except Exception as e:
        logger.error(f"ffmpegダウンロードエラー: {e}")
        logger.error("手動でffmpegをインストールするか、システムPATHに追加してください")
        return None


def check_ffmpeg() -> Tuple[bool, Optional[Path]]:
    """FFmpegの利用可能性をチェック（システム版 → ポータブル版の順）"""
    # まずシステム版をチェック
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            check=True
        )
        logger.info("システム版ffmpegを使用")
        return True, None
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # ポータブル版をチェック/ダウンロード
    portable_bin = download_ffmpeg_portable()
    if portable_bin:
        ffmpeg_exe = portable_bin / "ffmpeg.exe"
        try:
            subprocess.run(
                [str(ffmpeg_exe), "-version"], 
                capture_output=True, 
                check=True
            )
            logger.info("ポータブル版ffmpegを使用")
            return True, portable_bin
        except subprocess.CalledProcessError:
            pass
    
    return False, None


# グローバル変数でffmpegのパスを保持
FFMPEG_BIN_PATH: Optional[Path] = None

def get_ffmpeg_cmd(command: str) -> str:
    """ffmpegコマンドのフルパスを取得"""
    if FFMPEG_BIN_PATH:
        return str(FFMPEG_BIN_PATH / f"{command}.exe")
    return command


def get_audio_duration(file_path: Path) -> float:
    """音声ファイルの長さを秒で取得"""
    try:
        result = subprocess.run([
            get_ffmpeg_cmd("ffprobe"), "-v", "quiet", "-print_format", "json",
            "-show_format", str(file_path)
        ], capture_output=True, check=True, text=True)
        
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        raise FFmpegError(f"音声長取得エラー: {e}")


def compress_audio(
    input_path: Path, 
    output_path: Path, 
    bitrate: int = 24,
    start_time: Optional[float] = None,
    duration: Optional[float] = None
) -> None:
    """音声ファイルをOGG/OPUS形式で圧縮"""
    cmd = [
        get_ffmpeg_cmd("ffmpeg"), "-y", "-v", "quiet",
        "-i", str(input_path),
        "-vn",  # 動画ストリーム無効
        "-acodec", "libopus",
        "-b:a", f"{bitrate}k",
        "-ar", "16000",  # 16kHz
        "-ac", "1",      # モノラル
    ]
    
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    
    cmd.append(str(output_path))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"音声圧縮エラー: {e}")


def estimate_cost(duration_seconds: float) -> float:
    """転写コストを見積もり (USD)"""
    # Whisper-1: $0.006 per minute
    minutes = duration_seconds / 60.0
    return minutes * 0.006


def create_chunks(
    input_path: Path, 
    output_dir: Path, 
    chunk_duration: int,
    bitrate: int
) -> List[Tuple[Path, float, float]]:
    """大きなファイルをチャンクに分割"""
    total_duration = get_audio_duration(input_path)
    num_chunks = math.ceil(total_duration / chunk_duration)
    
    logger.info(f"ファイルを{num_chunks}個のチャンクに分割中...")
    
    chunk_files = []
    
    with tqdm(total=num_chunks, desc="チャンク作成", unit="chunk") as pbar:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            duration = min(chunk_duration, total_duration - start_time)
            
            chunk_path = output_dir / f"chunk_{i:03d}.ogg"
            compress_audio(
                input_path, 
                chunk_path, 
                bitrate=bitrate,
                start_time=start_time,
                duration=duration
            )
            
            chunk_files.append((chunk_path, start_time, duration))
            pbar.update(1)
    
    return chunk_files


def transcribe_with_retry(
    client: openai.OpenAI,
    file_path: Path,
    model: str = "whisper-1",
    language: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """指数バックオフ付きリトライでWhisper API呼び出し"""
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio_file:
                kwargs = {
                    "file": audio_file,
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                
                if language:
                    kwargs["language"] = language
                
                response = client.audio.transcriptions.create(**kwargs)
                return response.model_dump()
                
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 60  # 1分, 2分, 4分
                logger.warning(f"レート制限エラー。{wait_time}秒後にリトライ...")
                time.sleep(wait_time)
            else:
                raise WhisperError(f"レート制限エラー (最大リトライ回数に達しました): {e}")
                
        except openai.APIError as e:
            if attempt < max_retries - 1 and e.status_code >= 500:
                wait_time = (2 ** attempt) * 10  # 10秒, 20秒, 40秒
                logger.warning(f"サーバーエラー。{wait_time}秒後にリトライ...")
                time.sleep(wait_time)
            else:
                raise WhisperError(f"API エラー: {e}")
    
    raise WhisperError("最大リトライ回数に達しました")


def merge_transcriptions(chunk_results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    """複数チャンクの転写結果をマージ"""
    if not chunk_results:
        return {"text": "", "segments": []}
    
    merged_text = []
    merged_segments = []
    
    for result, offset in chunk_results:
        merged_text.append(result.get("text", "").strip())
        
        for segment in result.get("segments", []):
            # タイムスタンプにオフセットを加算
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += offset
            adjusted_segment["end"] += offset
            merged_segments.append(adjusted_segment)
    
    return {
        "text": " ".join(filter(None, merged_text)),
        "segments": merged_segments,
        "language": chunk_results[0][0].get("language", "unknown")
    }


def format_timestamp_srt(seconds: float) -> str:
    """SRT形式のタイムスタンプを生成"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_content(segments: List[Dict[str, Any]]) -> str:
    """SRT字幕ファイルの内容を生成"""
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp_srt(segment["start"])
        end_time = format_timestamp_srt(segment["end"])
        text = segment["text"].strip()
        
        # 長い行を分割 (42文字制限)
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 42:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        srt_lines.extend([
            str(i),
            f"{start_time} --> {end_time}",
            "\n".join(lines),
            ""
        ])
    
    return "\n".join(srt_lines)


def save_outputs(
    result: Dict[str, Any], 
    base_path: Path, 
    output_dir: Optional[Path] = None
) -> None:
    """転写結果を各種形式で保存"""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_output = output_dir / base_path.stem
    else:
        base_output = base_path.parent / base_path.stem
    
    # プレーンテキスト
    txt_path = base_output.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # JSON (生データ)
    json_path = base_output.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # SRT字幕
    srt_path = base_output.with_suffix(".srt")
    srt_content = create_srt_content(result["segments"])
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    logger.info(f"出力ファイル:")
    logger.info(f"  テキスト: {txt_path}")
    logger.info(f"  字幕: {srt_path}")
    logger.info(f"  JSON: {json_path}")


def process_file(
    file_path: Path,
    args: argparse.Namespace,
    client: openai.OpenAI
) -> None:
    """単一ファイルの処理メイン関数"""
    logger.info(f"処理開始: {file_path}")
    
    # 音声長取得とコスト見積もり
    try:
        duration = get_audio_duration(file_path)
        cost = estimate_cost(duration)
        logger.info(f"音声長: {duration/60:.1f}分, 推定コスト: ${cost:.3f}")
    except FFmpegError as e:
        logger.error(f"ファイル分析エラー: {e}")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 音声圧縮
        logger.info("音声圧縮中...")
        compressed_path = temp_path / "compressed.ogg"
        
        try:
            compress_audio(file_path, compressed_path, bitrate=args.bitrate)
            compressed_size_mb = compressed_path.stat().st_size / (1024 * 1024)
            logger.info(f"圧縮完了: {compressed_size_mb:.1f} MB")
        except FFmpegError as e:
            logger.error(f"音声圧縮エラー: {e}")
            return
        
        # チャンク分割の判定
        if compressed_size_mb > 24.5:  # 0.5MB のマージン
            logger.info("ファイルサイズが大きすぎるため、チャンク分割を実行...")
            chunk_files = create_chunks(
                file_path, temp_path, args.chunk_sec, args.bitrate
            )
        else:
            chunk_files = [(compressed_path, 0.0, duration)]
        
        # 転写実行
        logger.info("転写開始...")
        chunk_results = []
        
        with tqdm(total=len(chunk_files), desc="転写進行", unit="chunk") as pbar:
            for chunk_path, start_time, _ in chunk_files:
                try:
                    result = transcribe_with_retry(
                        client, chunk_path, 
                        model=args.model, 
                        language=args.language
                    )
                    chunk_results.append((result, start_time))
                    pbar.update(1)
                except WhisperError as e:
                    logger.error(f"転写エラー: {e}")
                    return
        
        # 結果マージ
        logger.info("結果マージ中...")
        final_result = merge_transcriptions(chunk_results)
        
        # 出力保存
        save_outputs(final_result, file_path, args.out_dir)
        logger.info("処理完了")


def main():
    """メイン関数"""
    global FFMPEG_BIN_PATH
    
    parser = argparse.ArgumentParser(
        description="OpenAI Whisper APIを使用した動画・音声転写ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s meeting.mp4
  %(prog)s --language ja --chunk-sec 600 long_video.mp4
  %(prog)s --out-dir ./transcripts *.mp4
        """
    )
    
    parser.add_argument(
        "files", 
        nargs="+", 
        type=Path,
        help="転写する動画・音声ファイル"
    )
    parser.add_argument(
        "--out-dir", 
        type=Path,
        help="出力ディレクトリ (デフォルト: 入力ファイルと同じディレクトリ)"
    )
    parser.add_argument(
        "--chunk-sec", 
        type=int, 
        default=900,
        help="チャンク分割時の長さ (秒, デフォルト: 900=15分)"
    )
    parser.add_argument(
        "--bitrate", 
        type=int, 
        default=24,
        help="音声ビットレート (kbps, デフォルト: 24)"
    )
    parser.add_argument(
        "--language", 
        type=str,
        help="音声言語 (例: ja, en) - 自動検出の場合は省略"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="whisper-1",
        help="Whisperモデル (デフォルト: whisper-1)"
    )
    parser.add_argument(
        "--api-key", 
        type=str,
        help="OpenAI APIキー (環境変数 OPENAI_API_KEY でも指定可能)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="詳細ログを表示"
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # FFmpeg確認
    ffmpeg_available, portable_path = check_ffmpeg()
    if not ffmpeg_available:
        logger.error("FFmpegが利用できません。")
        sys.exit(2)
    
    FFMPEG_BIN_PATH = portable_path
    
    # APIキー設定
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません。")
        logger.error("--api-key オプションまたは OPENAI_API_KEY 環境変数を設定してください。")
        sys.exit(1)
    
    client = openai.OpenAI(api_key=api_key)
    
    # ファイル存在確認
    valid_files = []
    supported_extensions = {".mp4", ".wav", ".mp3", ".m4a", ".mov", ".ogg"}
    
    for file_path in args.files:
        if not file_path.exists():
            logger.error(f"ファイルが見つかりません: {file_path}")
            continue
        
        if file_path.suffix.lower() not in supported_extensions:
            logger.warning(f"サポートされていない形式: {file_path}")
            continue
        
        valid_files.append(file_path)
    
    if not valid_files:
        logger.error("処理可能なファイルがありません。")
        sys.exit(1)
    
    # ファイル処理
    logger.info(f"{len(valid_files)}個のファイルを処理開始")
    
    for file_path in valid_files:
        try:
            process_file(file_path, args, client)
        except KeyboardInterrupt:
            logger.info("ユーザーによって中断されました。")
            sys.exit(1)
        except Exception as e:
            logger.error(f"予期しないエラー ({file_path}): {e}")
            continue
    
    logger.info("全ての処理が完了しました。")


if __name__ == "__main__":
    main() 