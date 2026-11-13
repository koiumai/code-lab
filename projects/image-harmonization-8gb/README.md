# 原寸を保つ画像調和プロトタイプ

貼り付け済み画像と物体マスクから、色・照明の低周波成分と接地影を調整する**実験用**ツールです。FLUX.2-klein-4Bを照明の参考画像を作るために使い、生成画像で元画像を置き換えません。

- 元の解像度・輪郭・細かな模様をなるべく残すため、元画素へ緩やかなRGB倍率を転写します。
- 影は元背景へ暗化率だけを転写します。影の強さは調整可能です。
- **物体マスクと影の許可領域の合計の外側は、元画素を厳密コピー**します。
- `--no-shadow` なら物体マスクの外を一切変更しません。物体外に新しい影を作ることと、物体外を一切変更しないことは同時に満たせません。

## 現時点の評価

公開サンプルで、旧来の調和モデルより有望な結果を確認しました。クマのぬいぐるみでは明るすぎる前景を抑え、元の毛並みを残して接地感を追加できました。人物でも接続した人型の影が生成されました。一方、影が濃すぎる、形が入力と合わない、seedで結果が変わる等の問題が残ります。

**RTX 2080実機では未検証です。** RTX A5000でNF4/FP16計算、FP32 VAE、GPU割当上限6.5GiBという構成を試しました。BF16/FP8はこの低VRAM経路では使いません。ただしGPU世代・ドライバ・OS・CUDAカーネルの違いまでは再現できません。動作保証や制作工程の完成品ではありません。

## 準備

Ubuntu24.04・Python3.12・PyTorch2.8.0+CUDA12.8のクリーン環境で、この手順から配布CLIのGPU通し実行まで確認しました（A40、公開2例）。Windows実機は未検証です。Windowsで利用できないCPUピーク計測は自動で省略し、推論を継続します。既存の業務用Python環境を上書きせず、仮想環境を作ってください。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CUDA12.8に対応したNVIDIAドライバが必要です。社内PCのドライバ更新やソフト導入は会社の手順に従ってください。CPU RAMは16GB以上推奨、初回の重み取得用に十分なディスク容量（環境を含め約40GBの空きが目安）を用意してください。画像はモデルのAPIへ送信せずローカル推論します。初回はHugging Faceから公開モデルをダウンロードします。

## 起動前チェック

仮想環境を有効にした状態で以下を実行できます。GPU・CUDA・空きVRAM・RAM・sm_75対応の情報を表示するだけで、設定変更やインストールは行いません。

```bash
python preflight.py
# 機械可読の診断結果
python preflight.py --json
```

警告なしでもRTX2080での推論成功を保証するものではありません。`nvidia-smi`で他プロセスがVRAMを占有していないかも確認してください。

## 実行

白い領域が物体、黒い領域が背景のグレースケールPNGマスクを、入力画像と同じ幅・高さで用意します。

```bash
python harmonize.py --image composite.png --mask object_mask.png --output result.png
```

厳密に物体マスクの内側だけを調整する場合:

```bash
python harmonize.py --image composite.png --mask object_mask.png --output relit.png --no-shadow
```

影を描いてよい領域を指定するとき:

```bash
python harmonize.py --image composite.png --mask object_mask.png \
  --shadow-mask shadow_allowed.png --output result.png \
  --prompt "Match the teddy bear lighting to the overcast scene; add a subtle soft contact shadow. Preserve its shape, fur and bow."
```

物体maskだけでも動かせるよう、既定では物体の下端周辺に楕円形の接地影許可領域を推定します。長く伸びる影を既定領域で切ってしまう場合があります。**床面や光源を理解した領域推定ではありません。** 長い影、斜面、壁、浮遊物体、複数物体では外れるため、書き出された影許可マスクを確認し、必要なら明示的なmaskを渡してください。許可領域を広げれば変更可能な背景も広がります。

既定の内部解像度は最大辺768です。最終出力は常に元画像のサイズですが、低解像度の照明参照を拡大して使うため、細かな照明・影の精度まで原寸相当になるわけではありません。

```bash
# 精細な照明参照を試す（8GBでは余裕が小さくなる）
python harmonize.py --image composite.png --mask object_mask.png --output result.png \
  --max-side 1024 --memory-gib 6.5 --seed 42 --shadow-strength 0.6
```

メモリ不足なら他のGPUアプリを閉じ、`--max-side 512`または768へ下げてください。`--memory-gib`はPyTorchの割当制限であり、CUDAコンテキスト等を含む総VRAM制限ではありません。8GB GPUに8GiBを割り当てる設定は推奨しません。

出力はPNGのみです。`result-assets/`に元画像、物体マスク、影許可マスク、許可領域の合計、生成参照、実行パラメータ・不変性検査結果を保存します。既存の出力や同名のassetsフォルダは上書きせず停止します。

重みをロードせず既存の参照画像で合成だけを試せます:

```bash
python harmonize.py --image composite.png --mask object_mask.png \
  --reference generated_reference.png --output test.png --no-shadow
```

## 品質上の限界

- 元の質感は倍率転写で保持しますが、明暗・色を変えるため画素値は変わります。過大な倍率は白飛び・黒つぶれを招きます。
- 生成参照と元物体の位置・形がずれると、照明転写にハローや不自然な暗さが生じます。3D形状・材質を推定した物理レンダーではありません。
- モデルは物体マスクそのものに拘束された編集をするわけではありません。最終合成が変更範囲を制限します。複数被写体ではプロンプトで対象を明確にしてください。
- 領域外不変はコードで検査できますが、自然な光源・色・影は目視で判断する必要があります。自動的な「品質合格」はありません。
- 影は暗化のみで表現するため、色付き間接光や反射、透過、複雑な床面には適しません。
- 新しいディテールを生成して元物体に足す方法ではなく、大きな照明変化の表現力と元の細部保持を交換しています。

## 検証・ライセンス

`EVALUATION.md`に測定値と比較結果を記載しています。モデルなしの単体テストは以下で実行できます。

```bash
python -m unittest discover -s tests -v
```

[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)の公式モデルカードはApache-2.0です。使用時点のモデル・依存ライブラリの利用条件を確認してください。この配布物に重みや第三者サンプル写真は含めません。公開サンプルの評価結果は一般の業務画像に対する保証ではなく、会社での導入可否は社内ルールに従ってください。

本ディレクトリの独自コードのライセンスは[LICENSE](LICENSE)、モデル・依存・参照元の情報は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
