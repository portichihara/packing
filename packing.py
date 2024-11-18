import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class PackingRobotAnalyzer:
    def __init__(self):
        """パッキングロボットデータ分析クラスの初期化"""
        self.data = None
        
    def generate_mock_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """テスト用のモックデータを生成"""
        np.random.seed(42)
        
        # 製品IDの生成
        product_ids = [f"PROD_{i:04d}" for i in range(n_samples)]
        
        # サイズデータの生成（LxWxH）
        sizes = np.random.normal(loc=[300, 200, 150], scale=[50, 30, 20], size=(n_samples, 3))
        sizes = np.clip(sizes, 50, 500)  # 現実的な範囲に制限
        size_strings = [f"{l:.0f}x{w:.0f}x{h:.0f}" for l, w, h in sizes]
        
        # 重量データの生成
        weights = np.clip(np.random.normal(loc=5, scale=2, size=n_samples), 0.1, 15)
        
        # パッケージングタイプ
        packaging_types = np.random.choice(
            ['標準段ボール', '緩衝材強化', 'エコパッケージ', '専用ケース'],
            size=n_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        # 作業位置データの生成
        positions = np.random.normal(loc=[500, 400, 300], scale=[100, 80, 60], size=(n_samples, 3))
        position_strings = [f"{x:.0f}x{y:.0f}x{z:.0f}" for x, y, z in positions]
        
        # コンベヤ速度の生成
        conveyor_speeds = np.clip(np.random.normal(loc=0.5, scale=0.1, size=n_samples), 0.2, 0.8)
        
        # ツールタイプの生成
        tool_types = np.random.choice(
            ['標準グリッパー', '吸引パッド', '広範囲グリッパー', '精密グリッパー'],
            size=n_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        # タイムスタンプの生成
        base_date = datetime.now() - timedelta(days=7)
        timestamps = [base_date + timedelta(
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        ) for _ in range(n_samples)]
        
        # データフレームの作成
        self.data = pd.DataFrame({
            'ProductID': product_ids,
            'Size_LxWxH_mm': size_strings,
            'Weight_kg': weights,
            'Packaging': packaging_types,
            'WorkPosition_xyz_mm': position_strings,
            'ConveyorSpeed_m_per_s': conveyor_speeds,
            'ToolType': tool_types,
            'Timestamp': timestamps
        })
        
        return self.data
    
    def analyze_product_clusters(self, n_clusters: int = 3) -> Dict:
        """製品サイズと重量に基づくクラスタリング分析"""
        if self.data is None:
            self.generate_mock_data()
            
        # サイズデータの解析
        size_data = np.array([list(map(float, size.split('x'))) 
                            for size in self.data['Size_LxWxH_mm']])
        
        # 特徴量の準備
        features = np.column_stack((size_data, self.data['Weight_kg']))
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # クラスタリング実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # クラスターごとの統計
        cluster_stats = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_stats[f'cluster_{i}'] = {
                'count': int(np.sum(mask)),
                'avg_size': size_data[mask].mean(axis=0).tolist(),
                'avg_weight': float(self.data['Weight_kg'][mask].mean()),
                'common_tool': self.data['ToolType'][mask].mode()[0],
                'common_packaging': self.data['Packaging'][mask].mode()[0]
            }
        
        return cluster_stats
    
    def analyze_workload_patterns(self) -> Dict:
        """作業負荷パターンの分析"""
        if self.data is None:
            self.generate_mock_data()
            
        self.data['hour'] = pd.to_datetime(self.data['Timestamp']).dt.hour
        
        hourly_stats = {
            'product_count': self.data.groupby('hour').size().to_dict(),
            'avg_conveyor_speed': self.data.groupby('hour')['ConveyorSpeed_m_per_s'].mean().to_dict(),
            'tool_usage': self.data.groupby(['hour', 'ToolType']).size().unstack(fill_value=0).to_dict()
        }
        
        return hourly_stats
    
    def optimize_packaging_recommendations(self) -> List[Dict]:
        """パッケージング最適化の推奨事項を生成"""
        if self.data is None:
            self.generate_mock_data()
            
        recommendations = []
        
        # サイズベースの分析
        size_data = np.array([list(map(float, size.split('x'))) 
                            for size in self.data['Size_LxWxH_mm']])
        volume = np.prod(size_data, axis=1)
        
        # 効率的なパッケージングパターンの特定
        efficiency_score = volume / self.data['Weight_kg']
        efficient_patterns = self.data[efficiency_score > np.percentile(efficiency_score, 75)]
        
        for _, pattern in efficient_patterns.head().iterrows():
            recommendations.append({
                'product_id': pattern['ProductID'],
                'packaging_type': pattern['Packaging'],
                'efficiency_score': float(efficiency_score[self.data['ProductID'] == pattern['ProductID']].iloc[0]),
                'tool_type': pattern['ToolType'],
                'conveyor_speed': float(pattern['ConveyorSpeed_m_per_s'])
            })
        
        return recommendations
    
    def visualize_work_positions(self) -> None:
        """作業位置の3Dプロット生成"""
        if self.data is None:
            self.generate_mock_data()
            
        positions = np.array([list(map(float, pos.split('x'))) 
                            for pos in self.data['WorkPosition_xyz_mm']])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(positions[:, 0], 
                           positions[:, 1], 
                           positions[:, 2],
                           c=self.data['Weight_kg'],
                           cmap='viridis')
        
        plt.colorbar(scatter, label='Weight (kg)')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_zlabel('Z Position (mm)')
        plt.title('Work Positions Distribution')
        plt.show()

def main():
    # 分析クラスのインスタンス化
    analyzer = PackingRobotAnalyzer()
    
    # モックデータの生成
    data = analyzer.generate_mock_data(n_samples=1000)
    print("モックデータ生成完了:", len(data), "件")
    
    # クラスタリング分析
    clusters = analyzer.analyze_product_clusters()
    print("\n製品クラスター分析結果:")
    for cluster, stats in clusters.items():
        print(f"\n{cluster}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 作業負荷パターン分析
    workload = analyzer.analyze_workload_patterns()
    print("\n時間帯別作業負荷:")
    print(f"最も忙しい時間帯: {max(workload['product_count'].items(), key=lambda x: x[1])[0]}時")
    
    # パッケージング最適化推奨
    recommendations = analyzer.optimize_packaging_recommendations()
    print("\nパッケージング最適化推奨:")
    for rec in recommendations[:3]:  # Top 3の推奨事項を表示
        print(f"ProductID: {rec['product_id']}")
        print(f"  効率スコア: {rec['efficiency_score']:.2f}")
        print(f"  推奨パッケージ: {rec['packaging_type']}")
        print(f"  推奨ツール: {rec['tool_type']}")
        print(f"  推奨コンベヤ速度: {rec['conveyor_speed']:.2f} m/s")
    
    # 作業位置の可視化
    analyzer.visualize_work_positions()

if __name__ == "__main__":
    main()