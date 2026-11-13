# 原寸を保つ画像調和プロトタイプ

貼り付け済み画像と物体マスクから、色・照明の低周波成分と接地影を調整する**実験用**ツールです。FLUX.2-klein-4Bを照明の参考画像を作るために使い、既定のgainモードでは生成画像で元画像を置き換えません。品質優先の実験用blendモードでは物体マスク内に限り生成参照を混合できます。

- 元の解像度・輪郭・細かな模様をなるべく残すため、元画素へ緩やかなRGB倍率を転写します。
- 影は元背景へ暗化率だけを転写します。影の強さは調整可能です。
- **物体マスクと影の許可領域の合計の外側は、元画素を厳密コピー**します。
- `--no-shadow` なら物体マスクの外を一切変更しません。物体外に新しい影を作ることと、物体外を一切変更しないことは同時に満たせません。


## 品質優先モードのローカル試作（未公開）

このコピーは調和品質を比較する実験版です。従来版・公開済みコードは変更しません。
`--object-mode` と `--object-strength` は物体だけに作用します。影の計算・強度は従来のままです。

- `gain`（既定）: 平滑化したRGB倍率を転写します。`--object-strength 1` は従来版互換です。
- `delta`: 平滑化した参照と元画像のRGB差分を加算します。倍率方式より色・明るさの変化を出せる場合がありますが、白飛び・黒つぶれも起こり得ます。
- `blend`: 生成参照そのものを物体マスク内で混合します。細部や形の再描画を許す代わりに調和が改善する可能性があります。輪郭の位置合わせは行わないため、元マスクと生成物体がずれると縁・残像が出ます。

`--object-strength` は0〜1（既定1）です。0なら物体画素を変えず、1なら各方式を最大適用します。
soft maskではさらにそのcoverageを掛けます。blendの強度1でもsoft maskの境界は完全置換ではありません。
物体マスクと影許可領域の合計の外側は、全モードで厳密コピーします。`--no-shadow`なら物体マスク外すべて不変です。

```bash
# 品質比較の開始点。生成参照は既存のものを再利用でき、GPU・通信不要です。
python harmonize.py --image composite.png --mask object_mask.png --reference reference.png \
  --output quality-blend.png --object-mode blend --object-strength 0.5
python harmonize.py --image composite.png --mask object_mask.png --reference reference.png \
  --output quality-delta.png --object-mode delta --object-strength 1
```

比較では同じ参照画像・物体マスク・影許可マスクを使ってください。参照を新規生成する場合はseedとpromptも揃えます。
今回の目的は細部を描き直すこと自体ではなく、物体全体の色・明るさを背景へ十分に馴染ませることです。gainは倍率0.5〜1.7の上限・下限を維持しています。参照自体の色・明るさが元画像から変わっていなければ、deltaやblendへ切り替えるだけでは改善しません。その場合は参照生成の指示・条件を見直す必要があります。

この変更だけで推論モデルやVRAM設定は変わりません。品質の優劣は入力画像ごとに目視確認が必要です。

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
- gain/deltaは主に照明を転写します。blendは生成ディテールも混ぜるため、照明変化の表現力と元の細部・同一性保持のトレードオフがあります。

## 検証・ライセンス

`EVALUATION.md`に測定値と比較結果を記載しています。モデルなしの単体テストは以下で実行できます。

```bash
python -m unittest discover -s tests -v
```

[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)の公式モデルカードはApache-2.0です。使用時点のモデル・依存ライブラリの利用条件を確認してください。この配布物に重みや第三者サンプル写真は含めません。公開サンプルの評価結果は一般の業務画像に対する保証ではなく、会社での導入可否は社内ルールに従ってください。

## 対象と照明を明示して調和を強める

既定は従来どおりです。物体全体の明るさ・ホワイトバランスが変わりにくいときは、対象と周囲の照明を明示した任意のprofileを使えます（英語の短い説明を推奨）。

```bash
python harmonize.py --image composite.png --mask object_mask.png --output relit.png \
  --lighting-profile photometric --subject "pink teddy bear" \
  --lighting "soft overcast daylight on the mountain road"
```

このprofileでは現在の物体の露出・色かぶりの変更を許し、背景全体の色調変更は要求しません。ただし**モデルが指示を守る保証はありません**。`reference.png`は背景も再生成された参考画像であり、完成画像ではありません。明る過ぎる参照をそのまま採用しないでください。最終PNGの背景は元画素へ戻しますが、物体の照明推定まで正しくなるわけではありません。raw参照の背景と元背景が大きく異なる場合、より具体的な`--prompt`で再生成するか、その結果を不採用にしてください。profileと`--prompt`は同時使用できません。

より強い見た目の変化を許す場合は、上のコマンドに `--object-mode blend --object-strength 0.75` を追加します。顔・模様・輪郭も変わり得るため、同一性が重要な商品・人物には特に注意してください。`delta`を強くして解決しようとすると暗部がつぶれる場合があります。

## 自動影許可領域を上下へ広げる

```bash
python harmonize.py --image composite.png --mask object_mask.png --output wider.png \
  --shadow-vertical-scale 1.5
```

既定の1.0に対し、楕円の中心と横幅を変えず縦半径を1.5倍にします。これは**物体マスクの膨張ではなく、影を変更してよい背景範囲の拡張**です。上下へ許可範囲が増えますが、正しい影の方向・長さを自動推定するわけではありません。広げ過ぎると背景の不要な暗化も許すため、`shadow-allowed.png` と `allowed-union.png`を必ず確認してください。明示的な`--shadow-mask`や`--no-shadow`とは併用せず、形を細かく指定するなら影マスクを用意します。倍率1.0の既定動作は変わりません。


## 背景拘束の別実験版

生成参照の背景露出・色が変わる問題に対し、生成途中から背景latentを拘束する [Klein背景拘束・実験版](../klein-background-anchor/) を別フォルダに用意しました。同じモデルcacheを再利用できます。この既存版のコード・既定動作は変更していません。公開2例で背景保持を確認しましたが、調和品質の保証はなく、A40測定のみです。実2080と配布wrapper入口のGPU通し実行は未検証です。
