### FXモデル（Milizeにやってもらうこと）

#### モデルリスク管理の準備
* 導入時検証として何を尋ねるか
    * 現行モデルの設定の再確認
    * 使用している特徴量と推定方法
    * PL計算方法
* 継続モニタリングとして何をするか
    * 勝率の確認（バックテスト）
    * PLの確認（パフォーマンスチェック）
* データを貰い直す。以前提供を受けたデータは第一生命さんのデータセットのような気が。。。

#### ベストではなくも良いので特徴量を固定する
* 特徴量を金庫から指示する。
* １パターンではなく複数パターン用意しておき常に全パターン計算する
* GAは設定が良いものか判断できないため外す

#### optunaは必ず入れる
in sampleでの最善を用いるという意味で。
> GAは設定が良いものか判断ができないため外す

#### PLを目的関数とする
大きく動くときの勝率を上げる（小さい変動は外しても良い）ことが目的となるはず。
全体としての勝率は低くても、PLにとって重要な局面での勝率が高ければ良い。

#### フラグ化案
* 4段階（大きく上昇、上昇、下降、大きく下降）のフラグ化
* 大きく上昇と上昇および下降と大きく下降の閾値は金庫から指示する。
* 大きく上昇／下降を目的関数の対象とする：PLを目的関数とするのと同様の効果
* 大きく上昇／下降のときだけポジションを取る（トレーディング戦略の変更）

#### トレーディング戦略
再考は今後の課題として３月までのMilizeさんR&Dのテーマからは外す？
> 例外的にフラグ化での大きく上昇／下降でのポジションについては可能であればやってもらう？






