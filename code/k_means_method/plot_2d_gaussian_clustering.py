
# k-means法 --------------------------------------------------------------------

# ch 4.3

# クラスタリング
# 更新推移の可視化

# 多次元混合ガウス分布
# 2次元の場合


# %%

# ディレクトリの設定 -------------------------------------------------------------

# ライブラリを読込
from pathlib import Path

# ワークスペースを取得
PROJECT_DIR = Path.cwd()
print(PROJECT_DIR)

# 書き出し先を設定
dir_path  = PROJECT_DIR.as_posix()
dir_path += '/figure/k_means_method/' # パスを指定
dir_path += 'plot_2d_gaussian_clustering/' # フォルダを指定
print(dir_path)


# %%

# 利用ライブラリ
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%

# 真のクラスタ数を指定
K_truth = 3

# K個の平均ベクトルを指定
mu = np.array(
    [[0.0, 5.0], 
     [5.0, -10.0], 
     [-10.0, -5.0]]
)

# K個の分散共分散行列を指定
sigma = np.array(
    [[[36.0, 10.0], [10.0, 25.0]], 
     [[9.0, -1.3], [-1.3, 16.0]], 
     [[25.0, -3.2], [-3.2, 16.0]]]
)

# 混合比率を指定
pi = np.array([0.45, 0.25, 0.3])

# %%

# 各軸の値を作成
x1_vals = np.linspace(
    np.min(mu[:, 0] - np.sqrt(sigma[:, 0, 0]) * 3.0), 
    np.max(mu[:, 0] + np.sqrt(sigma[:, 0, 0]) * 3.0), 
    num=300
)
x2_vals = np.linspace(
    np.min(mu[:, 1] - np.sqrt(sigma[:, 1, 1]) * 3.0), 
    np.max(mu[:, 1] + np.sqrt(sigma[:, 1, 1]) * 3.0), 
    num=300
)

# 作図用の点を作成
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# 作図用の点の形状を保存
x_dim = x1_grid.shape

# 計算用の点を作成
x_points = np.stack([x1_grid.flatten(), x2_grid.flatten()], axis=1)

# %%

# データ数を指定
N = 100

# 真のクラスタを生成
c_truth_onehot = np.random.multinomial(n=1, pvals=pi, size=N)

# 真のクラスタ番号を抽出
_, c_truth = np.where(c_truth_onehot == 1)

# データを生成
X = np.array(
    [np.random.multivariate_normal(mean=mu[j], cov=sigma[j], size=1).flatten() for j in c_truth]
)

# %%

# クラスタ数の初期値を指定
K = 10

# ランダムにクラスタを割り当て
c_onehot = np.random.multinomial(n=1, pvals=np.repeat(1, K)/K, size=N)

# クラスタ番号を抽出
_, c = np.where(c_onehot == 1)

# クラスタごとのデータ数を集計
G_num = np.sum(c_onehot, axis=0)

# クラスタの代表値(平均値)を計算
Z = np.array(
    [np.mean(X[c == j], axis=0) for j in range(K)]
)

# 目的関数(ノルムの2乗平均)を計算
J = np.array(
    [np.sum(np.linalg.norm(X[c == j] - Z[j], axis=1)**2) / N for j in range(K)]
)

# %%

# クラスタ数の初期値を指定
K = 10

# ランダムにクラスタの代表値を設定
Z = np.array(
    [np.random.uniform(low=X[:, 0].min(), high=X[:, 0].max(), size=K), 
     np.random.uniform(low=X[:, 1].min(), high=X[:, 1].max(), size=K)]
).T

# ノルムが最小のクラスタを割り当て
c = np.argmin(
    [np.linalg.norm(X - Z[j], axis=1) for j in range(K)], 
    axis=0
)

# クラスタごとのデータ数を集計
G_num = np.array(
    [np.sum(c == j) for j in range(K)]
)

# 目的関数(ノルムの2乗平均)を計算
J = np.array(
    [np.sum(np.linalg.norm(X[c == j] - Z[j], axis=1)**2) / N for j in range(K)]
)

# %%

# クラスタの最低割り当て数を指定
G_num_lower = 10

# 更新量の閾値を指定
threshold = 0.001

# 初期値を記録
trace_Z = [Z]
trace_c = [c]
trace_G = [G_num]
trace_J = [J]
# 繰り返し試行
cnt = 0
old_J_sum = np.sum(J)
while True:
    
    # 試行回数をカウント
    cnt += 1
    print('--- iter:' + str(cnt) + ' ---')
    
    # クラスタの代表値(平均値)を計算
    Z = np.array(
        [np.mean(X[c == j], axis=0) for j in range(K) if G_num[j] >= G_num_lower]
    )
    
    # クラスタ数を再設定
    K = len(Z)
    print('K=' + str(K))

    # ノルムが最小のクラスタを割り当て
    c = np.argmin(
        [np.linalg.norm(X - Z[j], axis=1) for j in range(K)], 
        axis=0
    )

    # クラスタごとのデータ数を集計
    G_num = np.array(
        [np.sum(c == j) for j in range(K)]
    )
    print('|G|=' + str(G_num))

    # 目的関数(ノルムの2乗平均)を計算
    J = np.array(
        [np.sum(np.linalg.norm(X[c == j] - Z[j], axis=1)**2) / N for j in range(K)]
    )
    J_sum = np.sum(J)
    print('J=' + str(J_sum.round(5)))
    
    # 更新値を記録
    trace_Z.append(Z)
    trace_c.append(c)
    trace_G.append(G_num)
    trace_J.append(J)
    
    # 更新量が閾値未満なら終了
    if abs(J_sum - old_J_sum) < threshold:
        break
    
    # 目的関数の値を保存
    old_J_sum = J_sum

# %%

# フレーム数を設定
frame_num = cnt

# カラーマップを指定
cm = plt.get_cmap('gist_rainbow')
colors = cm(np.arange(len(trace_Z[0])) / len(trace_Z[0]))

# クラスタ数の削減に応じて色を入れ替え
colors_lt = [colors]
for i in range(frame_num):
    colors = colors[np.argsort(-np.int16(trace_G[i] >= G_num_lower))] # 継続するクラスタ番号の色を前に出す
    colors_lt.append(colors)

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
fig.suptitle('k-means clustering', fontsize=20)

# 作図処理を関数として定義
def update(i):
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目の値を取得
    Z = trace_Z[i]
    c = trace_c[i]
    G_num = trace_G[i]
    J = trace_J[i]
    colors = colors_lt[i]
    
    # クラスタ数を再設定
    K = len(Z)
    
    # 2Dベクトルのクラスタを作図
    for j in range(K):
        G_j, = np.where(c == j) # クラスタjのデータインデック
        ax.plot([Z[j, 0].repeat(len(G_j)), X[G_j, 0]], 
                [Z[j, 1].repeat(len(G_j)), X[G_j, 1]], 
                color=colors[j], linewidth=1, linestyle=':', zorder=0) # 対応線
        ax.scatter(x=Z[j, 0], y=Z[j, 1], 
                   edgecolor=colors[j], facecolor='white', marker='s', s=100) # 代表値
        ax.scatter(x=X[G_j, 0], y=X[G_j, 1], 
                   color=colors[j], s=20, label=str(j+1)) # サンプルデータ
    ax.set_title('iter:' + str(i) + ', ' + 
                 'K=' + str(K) + '\n' + 
                 '$N=' + str(N) + '=(' + ', '.join(map(str, G_num)) + ')$\n' + 
                 '$J=' + str(np.sum(J).round(1)) + '=(' + ', '.join(map(str, J.round(1))) + ')$', loc='left')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid()
    ax.legend(title='cluster', loc='upper left')
    ax.set_aspect('equal')

# gif画像を作成
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=300)

# gif画像を保存
ani.save('k_means_gaussian_2d.gif')

# %%


