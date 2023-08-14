# 基本ライブラリ
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import collections

import detection_model

#利用するモデルをラジオボタンで選択
radio_model = st.sidebar.radio('利用するモデルを選んでください', detection_model.set_st_radio())

#利用するモデルをセット
model= detection_model.select_model(radio_model)

st.sidebar.table(data=detection_model.df_set())

st.title('柿ピー検出')

def predict(img):
        
    #ネットワークの準備
    #img : 画像データ
    #conf : 確率のMIN値
    
    results = model(img, conf=0.7)

    # 物体名を描画する
    font = ImageFont.truetype(font="ipaexg00401/ipaexg.ttf", size=60)  # フォントとサイズを指定する
    draw = ImageDraw.Draw(img)

    # CLASS_NAMES : 物体検出クラス
    CLASS_NAMES = ['柿の種', 'ピーナッツ']
    CLASS_COLORS = [(15, 15, 255), (0, 128, 0)]


    for pred in results:
        
        # pred : tensor型
        # box : 位置
        # cls : 物体検出クラス
        # conf : 確率
        
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
            #class_int 0:柿の種 1:ピーナッツ
            class_int = int(cls.numpy())
            
            
            # バウンディングボックスを描く
            # xmin : 左上
            # ymin : 左下
            # xmax : 右上
            # ymax : 右下
            
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=CLASS_COLORS[class_int], width=5)
            
            # 物体名と確率を描く
            # label : 物体名
            # label_with_prob : 物体名と確率
            

            class_name = CLASS_NAMES[class_int]
            label_with_prob = f'{class_name} {conf:.2f}'
            non, non, w, h = font.getbbox(label_with_prob)
            draw.rectangle([xmin, ymin, xmin+w, ymin+h], fill=CLASS_COLORS[class_int])
            draw.text((xmin, ymin), label_with_prob, fill="white", font=font)  # 物体名を描画する
    return results, img


# 画像のuploader
uploaded_image = st.file_uploader('画像をアップしてね！',type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    uploaded_image = Image.open(uploaded_image).convert('RGB')


    # 入力された画像に対して推論
    preds, draw_image = predict(uploaded_image)
    pred = preds[0]
    
    # kakinotane_count : 柿の種の数
    # nuts_count : ピーナッツの数
    
    kakinotane_nuts_count = collections.Counter(pred.boxes.cls.numpy())
    #kakinotane_count[0.0] が1個でもあればその数を出力、はない場合は0を出力
    kakinotane_count = kakinotane_nuts_count[0.0] if kakinotane_nuts_count[0.0] > 0 else 0
    nuts_count = kakinotane_nuts_count[1.0] if kakinotane_nuts_count[1.0] > 0 else 0
    
    kakinotane_list = [kakinotane_count, nuts_count]
    
    # 0が含まれているかどうか
    # if 0 not in kakinotane_list:
    #     #最大公約数
    #     gcd = math.gcd(*kakinotane_list)
    #     kakinotane_ratio, nuts_ratio = (int(i / gcd) for i in kakinotane_list)
    # else:
    #     kakinotane_ratio, nuts_ratio = kakinotane_count, nuts_count

    ratio_list = [ count_ / sum(kakinotane_list) if count_ > 0 else 0 for count_ in kakinotane_list] 

    #物体検出した画像を表示
    st.image(draw_image, use_column_width=True)

    st.subheader('個数')
    st.write(f'柿の種:{kakinotane_count}個' )
    st.write(f'ピーナッツ :{nuts_count}個' )

    st.subheader('割合')
    st.write('柿の種 : ピーナッツ' )
    st.write(f'{ratio_list[0]:.1%} : {ratio_list[1]:.1%}' )

    st.subheader('スピード')
    st.write(f'推論: {pred.speed["inference"]:.2f}ms')
