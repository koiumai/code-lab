# HINet 色・光調和 実験版

画像＋物体マスクから、前景の色味・明るさを背景になじませる小さなCLIです。元解像度を維持し、マスク外の画素は変更しません。汎用画像生成とは別方式で、柄・形を保った調和が目的です。

**新しい影は生成しません。** `--baseline` に既存の影付き画像を渡すと、マスク外だけをその画像からそのまま採用できます。物体内にはHINetの出力を100%使い、元画像との単純ブレンドはしていません。光源位置や立体的な陰影を物理的に再構築する保証はありません。

## 動作確認の範囲

- NVIDIA A40で、FP32・タイル推論・公式RAW_iHarmony4重みの安全な読み込みが成功。
- **RTX 2080実機では未検証です。** BF16やRTX 3000以降専用機能は使っていませんが、2080の速度・ドライバ互換性まで確認したものではありません。
- `--baseline`、出力の上書き拒否、出力合成ヘルパーはGPU実験後に追加し、CPUテスト9件で検証しました。配布セット全体を再度GPU実行したわけではありません。
- 2例の確認であり、全入力の品質や8GB以内の動作を保証しません。結果は `BENCHMARK.md`。

## セットアップ

Ubuntu、Python 3.10/3.11の独立した仮想環境を推奨します。既存の画像生成環境と依存関係を混ぜないでください。

```bash
python3 -m venv .venv
source .venv/bin/activate
# お使いのドライバに合う公式PyTorchとtorchvisionの組を先に導入
# 例: CUDA 12.1対応のドライバを使用する場合の互換ペア
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-inference.txt
```

上記PyTorch固定組は導入例であり、今回A40測定環境の再現を保証する指定ではありません。CPUのみなら公式CPU版を利用し、実行時に `--device cpu` を指定してください。

## 実行

```bash
python hinet_probe.py \
  --image composite.png --mask object-mask.png \
  --output result.png --download --device cuda --patch-size 256
```

- 画像とマスクは同一サイズ。マスクは白=物体、黒=保持。上流と同じ **画素値 >100** を物体とします。境界のソフト合成はしません。
- `--download` は未取得時に公式の固定リビジョンから重み約477MBを取得します。通常の学習済み重みを同梱していません。
- 元のcheckpointと、安全に抽出したmodel-onlyコピーを保持するため、重み用に約1GBの空き容量を確保してください。ダウンロード一時ファイルも使います。
- 出力はPNGと同名JSON。どちらかが既に存在すると上書きせず停止します。別の出力名を使ってください。
- `--patch-size 256` はタイルサイズの目安。メモリ不足時は128などに下げます。画像全体を256に縮小して出力する指定ではありません。
- 初回ダウンロード・モデル構築・読み込み時間は下記の推論秒数に含まれません。

既存の影を保持する場合:

```bash
python hinet_probe.py \
  --image original-composite.png --mask object-mask.png \
  --baseline existing-shadow-result.png \
  --output hinet-with-existing-shadow.png --download
```

`--image` は調和モデルが見る入力です。`--baseline` はモデルには渡さず、最後に物体の外側だけコピーします。位置・サイズが完全に一致する画像を使ってください。baseline内のマスク外編集は影以外もそのまま採用するため、使ってよい結果を指定してください。

## 安全なcheckpoint読み込み

公式checkpointは学習用optimizer/schedulerのPythonオブジェクトも含み、そのままの `weights_only=True` では読み込めません。本セットは次の手順で対応しています。

1. 元ファイル全体のSHA256を固定値と照合。
2. pickle本体のSHA256、既知の命令境界、残存GLOBALの限定リスト、スタック整合性を静的検証。
3. pickleを実行せずに学習用部分を除いた別ZIPを作成。
4. 抽出後も `torch.load(..., weights_only=True)` でロード。
5. 現在のモデルに存在しない既知の旧キー3個のみ除き、他のキーは `strict=True` で検証。

未知のcheckpointを読む汎用ツールではありません。安全ロードを無効にする変更は不要です。

## ライセンス・出典

- 公式: https://github.com/WindVChen/INR-Harmonization
- 元コードcommit: `40fa55a03ec5f2f832c48f3f98b8b83176b255dc`
- 重みの公式配布元: https://huggingface.co/spaces/WindVChen/INR-Harmon
- HF commit: `eab44a6a08e63cb129c93d1b7022c9c31c158326`
- モデル: `pretrained_models/Resolution_RAW_iHarmony4.pth`
- SHA256: `c5475d7a2a77260f8c9503b02e46a7b456bd1a7a7d7b0fc3c85ef534b473eef8`

元コードと、重みを同梱する公式HFリポジトリは、root LICENSEおよびREADMEでApache-2.0と宣言されています。確認範囲では重みの別条件・研究用途限定表記はありません。ただし、**重み個別のモデルカードでApache-2.0と明示した宣言はありません**。会社の利用審査では、この根拠と範囲を確認してください。公開ダウンロード可能という理由だけで商用利用を保証するものではありません。

上流のApache LICENSEを保持しています。画像・モデル重み・学習データセットはこの配布物に含みません。入力画像・比較画像・データセット自体の権利は別途確認が必要です。

## CPUテスト

```bash
python -m unittest discover -s tests -v
```

マスク外の厳密保持、baselineの分離、入力非破壊、寸法不一致、既存出力拒否、未知checkpoint拒否を確認します。モデル品質・GPU動作を検証するテストではありません。
