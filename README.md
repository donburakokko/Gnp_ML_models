# Gnp_ML_models
データ上げないでくださいね。データはUSB内に格納しているので、このディレクトリにはマジで入れないでね
それぞれの解析モデルごとにscriptを分けるのではなく、1つのscriptにまとめることにしました

ディレクトリ構成としては、statistics,modelで分けています

1. statisticsにはモデルの解析結果を載せますが、今の段階としてはそれぞれのモデル解析ごとに分けるようなディレクトリ構成にしたいと思います
　 ex) logstic, knn, nn, GBM, DecisionTreeみたいな感じを想定しています。その中に各データの解析結果が格納します
2. modelにはモデル構築すべてのsourceを載せたmodel.pyと評価指標系のモジュールをまとめたutils.py、および解析実行用のpredict.pyを格納します
3. 実行するには、git cloneもしくはdownload zipなどを活用してコード全体をダウンロードした後、modelディレクトリ上からpredict.py上のmain()関数を実行してください
 　つまり実行に差し当たって、ターミナルなどからpredict.pyをimportしてmain()を実行します。今回は、mainの引数として第1引数dataにデータ(csv)、第2引数yに目的変数のカラム名を入力してください
　 ※注意点として、目的変数以外が全て説明変数として投入されるので、必要な変数のみ含んだデータセットをdataに入力してください
 　下記にコード例を記述するので、参考にしてください
　
　 import main from predict
　 main(data = sample.csv, y = ["目的変数"])



4. 実行後は、statisticsディレクトリのそれぞれのモデル結果が格納されます。日付が記述されるので、どれが実行結果か分かると思います。
　 名前としては、(元データのcsv以前の名前部分+日付+解析モデル名).csvファイルとROCカーブの画像(元データのcsv以前の名前部分+日付+解析モデル名_ROC).pngとなります