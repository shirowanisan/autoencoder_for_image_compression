# autoencoder_for_image_compression
画像をオートエンコードするプログラム。言語はpython。

ソースコードの流れは以下のようになっています。
1. 画像をロードしてrgbの配列[1行, ピクセル数 * 3 (r, g b)列]にする。
2. rgb配列をオートエンコーダに入れる。
3. オートエンコーダの出力（rgbの配列）を画像の形式に戻す。
4. 3の画像を表示する。

## 環境構築
pyhonのバージョンは3.7.4で作成しました。
また、このソースコードは以下のライブラリを使っています。
```
$ pip install pillow
$ pip install numpy
$ pip install tensorflow
$ pip install keras
```
