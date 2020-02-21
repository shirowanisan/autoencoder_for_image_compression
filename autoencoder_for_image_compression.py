import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# input: 画像データ, 画像のサイズ
# output: r,g,bの配列
def get_rgb(image_data, image_size):
    r = []
    g = []
    b = []
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            current_r, current_g, current_b = image_data.getpixel((x, y))
            r.append(current_r)
            g.append(current_g)
            b.append(current_b)

    return r, g, b


# input: r,g,bを並べて一つの配列したもの, 画像のサイズ
# output: 画像を返す関数
def set_rgb(image_rgb, image_size):
    num_pixels = image_size[0] * image_size[1]
    red = []
    green = []
    blue = []
    for i in range(num_pixels * 3):
        if i < num_pixels:
            red.append(image_rgb[i])
        elif i < num_pixels * 2:
            green.append(image_rgb[i])
        else:
            blue.append(image_rgb[i])
    image = Image.new('RGB', image_size)
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            current_r = red[x + (y * image_size[0])]
            current_g = green[x + (y * image_size[0])]
            current_b = blue[x + (y * image_size[0])]
            image.putpixel((x, y), (current_r, current_g, current_b))

    return image


def main():
    # 画像をロード
    image = Image.open('image.png')
    # 画像のサイズを変更
    image_size = [30, 40]
    image = image.resize(image_size)

    # 画像のrgbを一つの配列にする
    r, g, b = get_rgb(image, image_size)
    rg = np.hstack((r, g))
    rgb = np.hstack((rg, b))

    # オートエンコーダの入力と出力を設定
    input_x = rgb / 255
    input_x = np.array([input_x])
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
    new_img = set_rgb(pred_rgb, image_size)

    # 画像データを出力
    new_img.show()


if __name__ == '__main__':
    main()
