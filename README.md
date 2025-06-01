# OpenAI Whisper CLI 転写ツール

OpenAI Whisper APIを使用して、長時間の動画・音声ファイルを効率的に転写するPythonコマンドラインツールです。

## 主な機能

- 音声圧縮による大幅なファイルサイズ削減 (約20分の1に圧縮)
- 大きなファイルの自動チャンク分割 (24MB制限対応)
- 複数出力形式: プレーンテキスト(.txt)、字幕(.srt)、生データ(.json)
- プログレスバー表示
- 指数バックオフによるリトライ機能
- コスト見積もり表示

## 必要なシステム要件

- Python 3.9以上
- FFmpeg (音声処理用)
- OpenAI APIキー

## インストール

1. リポジトリをクローンまたはダウンロード
2. 依存ライブラリをインストール:
   ```
   pip install -r requirements.txt
   ```
3. FFmpegをインストール:
   - Windows: https://ffmpeg.org/download.html からダウンロードして環境変数PATHに追加
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`

## 使用方法

### 基本的な使用方法

```bash
# 単一ファイル転写
python whisper_cli.py meeting.mp4

# 複数ファイル一括処理
python whisper_cli.py video1.mp4 video2.wav audio.mp3

# 出力ディレクトリ指定
python whisper_cli.py --out-dir ./transcripts meeting.mp4
```

### 高度なオプション

```bash
# 日本語音声の転写 (精度向上)
python whisper_cli.py --language ja meeting.mp4

# チャンク長を10分に設定 (メモリ制約がある場合)
python whisper_cli.py --chunk-sec 600 long_video.mp4

# 音声品質を上げる (ビットレート増加)
python whisper_cli.py --bitrate 48 high_quality_audio.wav

# 詳細ログ表示
python whisper_cli.py --verbose meeting.mp4
```

### 全オプション一覧

- `--out-dir PATH`: 出力ディレクトリ (デフォルト: 入力ファイルと同じディレクトリ)
- `--chunk-sec 900`: チャンク分割時の長さ (秒, デフォルト: 900=15分)
- `--bitrate 24`: 音声ビットレート (kbps, デフォルト: 24)
- `--language ja`: 音声言語 (例: ja, en) - 自動検出の場合は省略
- `--model whisper-1`: Whisperモデル (デフォルト: whisper-1)
- `--api-key sk-...`: OpenAI APIキー (環境変数でも指定可能)
- `--verbose`: 詳細ログを表示

## 環境変数設定

OpenAI APIキーを環境変数で設定することを推奨します:

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

### Windows (コマンドプロンプト)
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

### macOS/Linux
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

## 出力ファイル

各入力ファイルに対して以下3つのファイルが生成されます:

1. `<basename>.txt`: プレーンテキスト転写結果
2. `<basename>.srt`: 字幕ファイル (動画編集ソフトで利用可能)
3. `<basename>.json`: 生データ (タイムスタンプ、信頼度等を含む)

## 音声圧縮について

このツールは以下の設定で音声を圧縮し、帯域幅を大幅に削減します:

- コーデック: OPUS (Ogg コンテナ)
- サンプリングレート: 16kHz
- チャンネル: モノラル
- ビットレート: 24kbps (デフォルト)

この設定により、一般的なファイルサイズを約20分の1に削減できます。

## チャンク分割について

圧縮後のファイルが24MB (OpenAI制限 - 0.5MBマージン) を超える場合、
自動的に指定時間 (デフォルト15分) でチャンクに分割されます。

各チャンクは個別に転写され、最終的に時系列順で結合されます。

## コスト計算

Whisper-1モデルの料金: $0.006/分

コスト計算式: 音声時間(分) × $0.006

例: 90分の動画 = 90 × $0.006 = $0.54

## サポートする形式

- 動画: .mp4, .mov
- 音声: .wav, .mp3, .m4a, .ogg

## 制限事項

1. OpenAI API制限:
   - ファイルサイズ: 25MB (本ツールでは24MB+マージンで制御)
   - レート制限: デフォルト10リクエスト/分 (自動リトライで対応)

2. FFmpeg依存:
   - システムにFFmpegのインストールが必要

3. 音声品質:
   - 16kHz/24kbpsに圧縮するため、高音質が必要な場合は--bitrateオプションで調整

## トラブルシューティング

### FFmpegエラー
```
FFmpegが見つかりません。インストールしてPATHに追加してください。
```
→ FFmpegをインストールし、環境変数PATHに追加してください。

### APIキーエラー
```
OpenAI APIキーが設定されていません。
```
→ 環境変数OPENAI_API_KEYを設定するか、--api-keyオプションを使用してください。

### レート制限エラー
```
レート制限エラー。XX秒後にリトライ...
```
→ 自動的にリトライされます。頻繁に発生する場合はAPIプランの確認を推奨します。

### メモリ不足
チャンク長を短く設定してください:
```bash
python whisper_cli.py --chunk-sec 300 large_file.mp4
```

## ライセンス

このツールはMITライセンスの元で提供されています。
OpenAI APIの利用にはOpenAIの利用規約が適用されます。 
