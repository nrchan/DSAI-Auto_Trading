# DSAI-Auto_Trading

## 使用方式
跟原本規定的一樣
```
//安裝
pip install -r requirements.txt

//執行
python trader.py --training <training file> -- testing <testing file> --output <output file>
```

## 想法
### 觀察
先把原始的資料畫出來看看。

![raw data](images/raw-data.png)

看起來，一個大一點的週期可以超過 100 天。但是因為我們的目標只要 20 天，所以目前猜測，之後的模型在預測的當下不用看到太以前的資料。

![raw data 20](images/raw-data-20.png)

實際上把隨機某 20 天的資料畫出來，可以發現在短線內實在是沒有什麼肉眼可見的規律。

如果要單用之前的資料就一次預測未來的 20 筆資料，似乎有點強人所難。所以打算讓模型看一定長度的資料後，預測下一天的開盤價就好。
感覺大概是這樣：
```
[t0, t1, ... , tn  ] => [tn+1]
[t1, t2, ... , tn+1] => [tn+2]
[t2, t3, ... , tn+2] => [tn+3]
```