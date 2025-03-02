# PDF Extractor

PDFからテキスト、表、画像を抽出し、マークダウン形式に変換するツールです。

## 機能

- PDFからテキストを抽出し、マークダウン形式に変換
- PDFから表を抽出し、マークダウン形式に変換
- PDFからすべての画像を抽出し、個別ファイルとして保存
- 元のPDFレイアウト情報を保持したままマークダウン形式に変換
- ページ指定による部分的な抽出
- 表の自動検出と変換
- 文書構造の保持（見出し、段落など）
- 抽出した画像をマークダウンから参照

## インストール

### 方法1: 通常のインストール

1. リポジトリをクローン:
```
git clone https://github.com/yourusername/pdf_extractor.git
cd pdf_extractor
```

2. 依存ライブラリのインストール:
```
pip install -r requirements.txt
```

注意: tabula-pyにはJavaのインストールが必要です。

### 方法2: Dockerを使用する方法

1. Dockerイメージをビルド:
```
docker-compose build
```

2. コンテナを実行:
```
docker-compose run pdf-extractor [引数]
```

例えば:
```
docker-compose run pdf-extractor sample.pdf -o result.md
```

または、docker-compose.ymlの設定を変更して実行:
```yaml
command: "sample.pdf -o result.md --layout"
```

## 使用方法

### 基本: PDFをマークダウンに変換（表も統合）

デフォルトでは、PDFの全コンテンツをマークダウン形式に変換し、表もテキストフローに統合します:
```
python src/extractor.py path/to/your/file.pdf
```

ファイルに保存:
```
python src/extractor.py path/to/your/file.pdf -o output.md
```

特定のページをマークダウンとして抽出:
```
python src/extractor.py path/to/your/file.pdf --pages "1-5" -o output.md
```

### レイアウト情報を保持したマークダウン変換（新機能）

レイアウト情報を保持したままマークダウンに変換:
```
python src/extractor.py path/to/your/file.pdf --layout -o output_with_layout.md
```

この機能は以下を保持します:
- フォントサイズと太字情報に基づく見出しの検出
- テキストの位置情報に基づくレイアウト保持
- 表の構造化と正確な配置

### 画像の抽出と保存

PDFから画像のみを抽出して保存する場合:
```
python src/extractor.py path/to/your/file.pdf --images
```

画像の保存先ディレクトリを指定する場合:
```
python src/extractor.py path/to/your/file.pdf --images --image-dir my_images
```

画像を抽出せずにマークダウン変換する場合（デフォルトでは画像も抽出します）:
```
python src/extractor.py path/to/your/file.pdf --no-images
```

抽出された画像はマークダウン内で自動的に参照され、表示されます。例:
```markdown
## Extracted Images

![Image 1](extracted_images/image_12345.png)

![Image 2](extracted_images/image_67890.jpg)
```

### テキストのみを抽出

マークダウン形式なしでプレーンテキストのみを抽出:
```
python src/extractor.py path/to/your/file.pdf --text
```

テキストをファイルに保存:
```
python src/extractor.py path/to/your/file.pdf --text -o output.txt
```

### 表のみをマークダウンとして抽出

PDFから表のみを抽出:
```
python src/extractor.py path/to/your/file.pdf --tables
```

特定ページから表を抽出してファイルに保存:
```
python src/extractor.py path/to/your/file.pdf --tables --pages "1,3,5-7" -o tables.md
```

## Dockerでの実行例

```
# ヘルプを表示
docker-compose run pdf-extractor

# PDFをマークダウンに変換
docker-compose run pdf-extractor /data/sample.pdf -o /data/output.md

# レイアウト情報を保持したまま変換
docker-compose run pdf-extractor /data/sample.pdf --layout -o /data/output_layout.md

# 表のみを抽出
docker-compose run pdf-extractor /data/sample.pdf --tables -o /data/tables.md
```

注意: Docker内では、現在のディレクトリが `/data` にマウントされています。

## 主な特徴

- 多様なPDFファイルからのテキスト抽出
- 高精度な表の検出とマークダウン形式への変換
- PDFからすべての画像を抽出して個別ファイルとして保存
- レイアウト情報を保持したPDFからマークダウンへの変換
- グリッドベースと非グリッドベースの両方の方法による賢い表検出アルゴリズム
- 文書フローにシームレスに統合された表形式
- 空の行/列をクリーンアップする表構造の最適化処理
- マークダウンの見出しと段落による文書構造の忠実な再現
- 柔軟なページ指定による部分的な抽出
- 抽出したコンテンツの便利なファイル保存機能
- 簡単なナビゲーションのための出力内のページマーカー
- 自動画像検出・保存とマークダウンからの参照リンク生成

## 依存ライブラリ

- PyPDF2 - PDF解析とテキスト抽出
- tabula-py - 表抽出 (※Javaのインストールが必要)
- pandas - 表データの処理とフォーマット
- tabulate - マークダウン表の生成
- pdfminer.six - PDFのレイアウト解析
- Pillow - 画像処理と変換

## インストール要件

- Python 3.8 以上
- Java Runtime (tabula-pyに必要)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については [LICENSE](LICENSE) ファイルを参照してください。