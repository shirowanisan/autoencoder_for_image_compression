from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# input: 画像データ, 画像のサイズ
# output: r,g,bの配列
def getRGB(rgb_img, size):
    r = []
    g = []
    b = []
    for y in range(size[1]):
        for x in range(size[0]):
            current_r, current_g, current_b = rgb_img.getpixel((x, y))
            r.append(current_r)
            g.append(current_g)
            b.append(current_b)

    return r, g, b

# input: r,g,bを並べて一つの配列したもの, 画像のサイズ
# output: 画像を返す関数
def setRGB(rgb, size):
    num_pixels = size[0] * size[1]
    r = []
    g = []
    b = []
    for i in range(num_pixels * 3):
        if i < num_pixels:
            r.append(rgb[i])
        elif i < num_pixels * 2:
            g.append(rgb[i])
        else:
            b.append(rgb[i])
    tmp_img = Image.new('RGB', size)
    for y in range(size[1]):
        for x in range(size[0]):
            current_r = r[x + (y * size[0])]
            current_g = g[x + (y * size[0])]
            current_b = b[x + (y * size[0])]
            tmp_img.putpixel((x,y ), (current_r, current_g, current_b))

    return tmp_img

if __name__ == '__main__':
    # 画像をロード
    img = Image.open('img.jpg')
    # 画像のサイズを変更
    size = [30, 40]
    img = img.resize(size)

    # 画像のrgbを一つの配列にする
    r, g, b = getRGB(img, size)
    rg = np.hstack((r, g))
    rgb = np.hstack((rg, b))
    
    # オートエンコーダの入力と出力を設定
    # (Kerasがサンプル点1個では学習できなかったので、同じサンプル点を2個入れている)
    input_x = rgb / 255
    input_x = np.vstack((input_x, rgb / 255))
    output_y = input_x

    # オートエンコーダのモデルを定義
    dim_x = input_x.shape[1]
    model = Sequential()
    model.add(Dense(512, activation='sigmoid', input_dim=dim_x))
    model.add(Dense(dim_x, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # 学習
    model.fit(input_x, output_y, epochs=100, batch_size=1)

    # 予測
    pred = model.predict(input_x)
    
    # 予測の出力結果をrgbに戻す
    pred_rgb = pred[0]
    pred_rgb = pred_rgb * 255

    # rgbの配列を画像データに変える
    new_img = setRGB(pred_rgb, size)

    # 画像データを出力
    new_img.show()
