> 既存の [image-harmonization-8gb](../image-harmonization-8gb/) と同じKlein重みを使う、独立した背景拘束の実験版です。既存版のファイルや既定動作は変更しません。

# Klein 背景拘束・実験版

既存FLUX.2-klein-4Bを使い、生成途中の各stepで背景latentを元画像に対応するノイズ状態へ戻します。最後に背景を戻すだけの方式と違い、生成中にも背景を拘束します。

**公開2例で背景保持の改善を確認した実験候補です。万能な調和品質は保証しません。** クマは暗くなりすぎる傾向があるため、既定は`--blend 0.5`です。元画像と生成結果を物体内で半分ずつ混合します。0なら物体は元画像、1なら生成結果を全適用。顔・柄・細部や輪郭が変わる場合があります。

## 既存環境をそのまま使う

これまでのKlein版が動く仮想環境を有効化して、このフォルダで実行してください。モデルは同じ`black-forest-labs/FLUX.2-klein-4B`で、**同じHugging Face cacheがあればモデル重みの再ダウンロードは不要**です。別venvでもcache場所が同じなら再利用できます。HF_HOME等を変更した場合、cacheが不完全な場合、revisionを変更した場合は追加取得されます。cache確認等の通信は発生し得ますが、入力画像は外部送信しません。

動作確認に使った構成はPython3.12、PyTorch2.8.0+cu128、diffusers0.43.0、transformers5.22.2、accelerate1.15.0、bitsandbytes0.50.2です。既存環境の不要な再インストール・ドライバ変更はしないでください。requirements.txtは依存一覧です。

**この背景拘束経路はA40で検証しましたが、RTX2080実機では未検証です。** GPU測定は同一backendでanchor有無を比較するpaired probeを使いました。配布用wrapperはCPU単体・統合テストで確認しており、wrapperを入口とした通しGPU実行は未実施です。 NF4/FP16、FP32 VAEを使い、BF16/FP8は使いません。既定の6.5GiBはPyTorch割当上限でありdriverを含む総VRAM保証ではありません。必要なら`--max-side 512`へ下げてください。

## 実行

```bash
python harmonize_anchor.py --image composite.png --mask object_mask.png --output result.png \
  --prompt "Relight the pasted object to match the existing background lighting and exposure. Keep the background colors and brightness unchanged. Preserve the object silhouette and natural materials." \
  --blend 0.5
```

promptは対象を明確に書くほど指示しやすくなります。たとえば`the pink teddy bear`や、背景の`soft overcast daylight`等を具体化してください。日本語promptの品質は未検証です。

- maskは入力画像と同寸法のグレースケールPNG。白が編集対象、黒が固定。0/1と0/255を受け付けます。
- **最終PNGのmask外は元画像と完全一致**します。
- `--baseline existing_shadow.png`を指定するとmask外をこの既存画像からコピーします。影を新規生成せず、既存影を固定できます。baselineは元画像と同寸法にしてください。
- 影も新たに生成したい場合、物体外の地面までmaskに含める必要があります。ただし今回の比較は物体maskで検証したもので、拡張maskの影品質は未評価です。
- 内部最大辺768、seed42、4stepsが既定。`--max-side 512`や`--seed 123`で変更できます。
- PNG出力と`result-assets/`を保存します。既存ファイル・assetsは上書きせず停止します。
- `result-assets/reference.png`は生成参照、`result.png`はblendと厳密な背景コピーを施した最終結果です。参照自体はVAE由来の変化・境界の影響が残るためmask外完全一致ではありません。
- `--reference existing_reference.png`は**生成済み参照を混合するだけ**でGPUを使わず、背景拘束も新たに行いません。既存の明るすぎる参照を直す機能ではありません。

## 実装・利用条件

これは独自の単純な背景latent projectionであり、**LanPaintの実装ではありません**。LanPaintコードは同梱していません。Diffusers公式APIを呼び出す自作callbackです。pixel空間の厳密な条件付き生成を数学的に保証する方式ではありません。

自作コードはApache-2.0、モデルも別途Apache-2.0。第三者依存・モデル・入力画像の条件を置き換えるものではありません。LICENSEおよびTHIRD_PARTY_NOTICES.md参照。重み・写真・生成例は同梱しません。
