# 基本ライブラリ
import streamlit as st
from detection import model # detection.py からネットワークの定義を読み込み
from PIL import Image, ImageDraw, ImageFont



#自作データーセットを利用して学習したデータ

# YOLOv8モデルをもとに推論する
def predict(img):

    #ネットワークの準備
    #img : 画像データ
    #conf : 確率のMIN値

    results = model(img, conf=0.7)
    #物体名を描画する
    font = ImageFont.truetype(font="ipaexg00401/ipaexg.ttf", size=60)  # フォントとサイズを指定する
    draw = ImageDraw.Draw(img)

    #class_names : 物体検出クラス
    class_names = ['柿の種', 'ピーナッツ']


    for pred in results:

        # pred : tensor型
        # box : 位置
        # cls : 物体検出クラス
        # conf : 確率
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):

            # '''
            # バウンディングボックスを描く
            # xmin : 左上
            # ymin : 左下
            # xmax : 右上
            # ymax : 右下
            # '''
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=5)
            
            # '''
            # 物体名と確率を描く
            # label : 物体名
            # label_with_prob : 物体名と確率
            # '''
            label = class_names[int(cls.numpy())]
            label_with_prob = f'{label} {conf:.2f}'
            w, h = font.getsize(label_with_prob)
            draw.rectangle([xmin, ymin, xmin+w, ymin+h], fill='red')
            draw.text((xmin, ymin), label_with_prob, fill="white", font=font)  # 物体名を描画する
    return results, img


#画像のuploader
uploaded_image = st.file_uploader('画像をアップしてね！',type=['png', 'jpg', 'gif', 'jpeg'])
if uploaded_image is not None:
    uploaded_image = Image.open(uploaded_image).convert('RGB')


    # 入力された画像に対して推論
    preds, draw_image = predict(uploaded_image)
    # pred = preds[0]
    # '''
    # kakinotane_count_ : 柿の種の数
    # nuts_count_ : ピーナッツの数
    # '''
    # kakinotane_count = collections.Counter(pred.boxes.cls.numpy())
    # #kakinotane_count[0.0] が1個でもあればその数を出力、はない場合は0を出力
    # kakinotane_count_ = kakinotane_count[0.0] if kakinotane_count[0.0] > 0 else 0
    # nuts_count_ = kakinotane_count[1.0] if kakinotane_count[1.0] > 0 else 0
    
    # kakinotane_list = [kakinotane_count_, nuts_count_]
    
    # #0が含まれているかどうか
    # if 0 not in kakinotane_list:
    # # 最大公約数
    #     gcd = math.gcd(*kakinotane_list)
    #     kakinotane_ratio_, nuts_ratio_= (int(i / gcd) for i in kakinotane_list)
    # else:
    #     kakinotane_ratio_, nuts_ratio_ = kakinotane_count_, nuts_count_
    st.image(draw_image, caption='Drawn Image', use_column_width=True)