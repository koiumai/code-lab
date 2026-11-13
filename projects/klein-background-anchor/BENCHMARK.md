# 背景拘束の測定

2026-11-13、公開libcomサンプル2例。A40 / Ubuntu24.04、最大辺768、4steps、seed42、NF4 FP16 transformer/text encoder、FP32 VAE、PyTorch割当上限6.5GiB。同じ初期noiseでanchor無/有を比較しました。

| サンプル | allocated peak GiB | reserved peak GiB | driver sampled peak MiB | anchor無 / 有 推論秒 |
|---|---:|---:|---:|---:|
| bear |3.336|4.971|5425|3.106 / 2.881|
| woman |3.907|6.330|6817|3.157 / 3.098|

メモリpeakは重みロードと両modeを含むプロセス全体値で、anchor単独の値ではありません。driverは0.5秒間隔なので瞬間peakを取り逃す可能性があります。推論秒はモデルロード・初回download時間を含みません。

物体からモデル解像度で32px以上離れた背景の、元モデル入力とのMAE（255階調）:

| サンプル | anchor無 | anchor有 |
|---|---:|---:|
| bear |17.67|3.90|
| woman |19.68|1.31|

これは背景変化の抑制を測る数値で、全体の調和品質スコアではありません。このprobeでは独立したKlein VAE往復画像を保存していないため、残差のすべてをVAE由来と断定できません。最終合成では元画素をコピーし、mask外変更0を確認しました。

クマは暗くなりすぎる傾向があるため、配布wrapperではblend0.5を初期値としています。人物ではより穏やかな露出・色変化でした。2例の観察であり、未知画像の品質保証ではありません。

GPU検証は同一backendのpaired probeです。配布wrapperはCPU単体・統合テストで確認し、wrapper入口の通しGPU実行は未実施です。A40の割当制限はRTX2080実機検証の代わりにはなりません。
