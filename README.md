# Packing Robot Analyzer

### Overview
Packing Robot Analyzerは、パッキングロボットの作業データを分析するPythonプログラムです。このプログラムは以下の機能を提供します。

1. **モックデータの生成**  
   サンプルのパッキングロボットデータを自動生成します。

2. **製品クラスタリング分析**  
   製品サイズや重量に基づいてクラスタリングを行い、各クラスターの統計情報を生成します。

3. **作業負荷パターンの分析**  
   時間帯別の作業負荷を解析し、平均コンベヤ速度やツールの利用パターンを可視化します。

4. **パッケージング最適化推奨**  
   サイズ・重量データから効率的なパッケージングパターンを推奨します。

5. **作業位置の可視化**  
   3Dプロットを用いて作業位置データを可視化します。

---

### Installation

#### 必要要件
- Python 3.8以上
- 必要なライブラリ:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

#### インストール手順
1. このリポジトリをクローンします。
   ```bash
   git clone https://github.com/your-repo/packing-robot-analyzer.git
   cd packing-robot-analyzer
   ```
2. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

#### プログラムの実行
以下のコマンドを実行して、プログラムを開始します。
```bash
python packing_robot_analyzer.py
```

#### 主な出力
1. モックデータの生成件数
2. 製品クラスタリング分析の結果
3. 時間帯別の作業負荷統計
4. パッケージング最適化推奨事項
5. 作業位置の3Dプロット

---

### Functions

#### 1. `generate_mock_data(n_samples=1000)`
ランダムなモックデータを生成します。

#### 2. `analyze_product_clusters(n_clusters=3)`
サイズと重量を基に製品をクラスタリングし、各クラスターの統計情報を出力します。

#### 3. `analyze_workload_patterns()`
時間帯別の作業負荷、ツール使用状況、平均コンベヤ速度を解析します。

#### 4. `optimize_packaging_recommendations()`
効率スコアに基づき、最適なパッケージングタイプとツールを推奨します。

#### 5. `visualize_work_positions()`
3Dプロットを用いて作業位置を可視化します。

---

### Examples

#### クラスタリング分析結果
```plaintext
cluster_0:
  count: 340
  avg_size: [320.5, 210.3, 160.4]
  avg_weight: 5.2
  common_tool: 標準グリッパー
  common_packaging: 緩衝材強化
```

#### 時間帯別の作業負荷
```plaintext
最も忙しい時間帯: 14時
```

#### パッケージング最適化推奨
```plaintext
ProductID: PROD_0023
  効率スコア: 120.5
  推奨パッケージ: エコパッケージ
  推奨ツール: 吸引パッド
  推奨コンベヤ速度: 0.65 m/s
```

---

### Visualization
- 作業位置の3Dプロットは、作業の分布状況を視覚的に確認できます。

