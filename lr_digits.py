#%%
# datasetを読み込む
from sklearn.datasets import load_digits
 
# load_digitsの引数でクラス数を指定
# 2なら0と1, 3なら0と1と2が書かれたデータのみに絞られる
# 最大は10で0から9となる
digits = load_digits(10)
#%%
# dataにデータが入ってる
print(digits.data)

#%%
# 正解ラベルはtargetに入っている
print(digits.target)


#%%
# データの形
# データ1件あたり、8x8=64の特徴が配列(numpyのndarray)となっていて
# データ件数が1797件分ある
print(digits.data.shape)

#%%
# 今回は1500件を学習データ、残りの297件をテストデータにする
train_X = digits.data[:1500]
train_y = digits.target[:1500]
 
test_X = digits.data[1500:]
test_y = digits.target[1500:]
print(type(test_X))
#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()   #ロジスティック回帰
# fit関数で学習を行う
lr.fit(train_X, train_y)
 
# predict関数で予測を行う
pred = lr.predict(test_X) 
print(pred)
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, pred, labels=digits.target_names)

#%%
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([1, 4]))]
onx = convert_sklearn(lr, initial_types=initial_type)
with open("lr_digits.onnx", "wb") as f:
    f.write(onx.SerializeToString())

#%%
import onnx
import onnx.helper

#model = onnx.load('lr_digits.onnx')
model = onnx.load('lr_digits.onnx')
print(onnx.helper.printable_graph(model.graph))

#%% 
# digitの個別画像の確認 
import matplotlib.pyplot as plt
digits = load_digits(10)
digits_df = pd.DataFrame(digits.data)
plt.imshow(digits_df.loc[4,:].values.reshape(8,8))
print(digits_df.loc[4,:])
#%% [markdown]
# ```
# markdown
# ```
# # test

#%%
import onnxruntime as rt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits(10)
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
sess = rt.InferenceSession("lr_digits.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name)
print(label_name)
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)


#%%
