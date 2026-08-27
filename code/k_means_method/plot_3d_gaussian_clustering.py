
# k-means法 --------------------------------------------------------------------

# ch 4.3

# クラスタリング
# 更新推移の可視化

# 多次元混合ガウス分布
# 3次元の場合


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
dir_path += 'plot_3d_gaussian_clustering/' # フォルダを指定
print(dir_path)


# %%

# ライブラリを読込
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# %%

# トイデータの作成 ---------------------------------------------------------------

### 生成分布の設定 -----

# 次元数を指定:(固定)
D = 3

# 真のクラスタ数を指定
K_truth = 3

# K個の平均ベクトルを指定
mu_kd = np.array(
    [[0.0, 2.5, 5.0], 
     [5.0, -5.0, -5.0], 
     [-10.0, 0.0, -2.5]]
)

# K個の分散共分散行列を指定
sigma2_kdd = np.array(
    [[[36.0, 0.0, 10.0], [0.0, 4.0, -5.0], [10.0, -5.0, 25.0]], 
     [[9.0, 0.0, -1.3], [0.0, 9.0, 0.0], [-1.3, 0.0, 16.0]], 
     [[25.0, -5.0, -3.2], [-5.0, 9.0, -1.2], [-3.2, -1.2, 16.0]]]
)

# 混合比率を指定
pi_k = np.array([0.45, 0.25, 0.3])


# %%

### 生成分布の計算 -----

# x軸の範囲を設定
u       = 5.0
sgm_num = 2.0
x_0_min, x_1_min, x_2_min = np.min(
    [mu_kd[k] - sgm_num * np.sqrt(np.diag(sigma2_kdd[k])) for k in range(K_truth)], 
    axis=0
)
x_0_max, x_1_max, x_2_max = np.max(
    [mu_kd[k] + sgm_num * np.sqrt(np.diag(sigma2_kdd[k])) for k in range(K_truth)], 
    axis=0
)
x_0_min, x_1_min, x_2_min = np.floor(np.array([x_0_min, x_1_min, x_2_min]) /u)*u # u単位で切り下げ
x_0_max, x_1_max, x_2_max = np.ceil(np.array([x_0_max, x_1_max, x_2_max]) /u)*u  # u単位で切り上げ
print(x_0_min, x_0_max)
print(x_1_min, x_1_max)
print(x_2_min, x_2_max)

# x軸の値を作成
x_0_vec = np.linspace(start=x_0_min, stop=x_0_max, num=251)
x_1_vec = np.linspace(start=x_1_min, stop=x_1_max, num=251)
x_2_vec = np.linspace(start=x_2_min, stop=x_2_max, num=101) # 等高線の数を指定

# 格子点を作成
x_0_grid, x_1_grid = np.meshgrid(x_0_vec, x_1_vec) # 平面座標

# 形状を保存
grid_shape = x_0_grid.shape
grid_size  = x_0_grid.size

# %%

# 受け皿を初期化
dens_lt = []

# z軸ごとに計算
for i in range(len(x_2_vec)):

    # 座標を作成
    x_arr = np.stack(
        [x_0_grid.flatten(), x_1_grid.flatten(), x_2_vec[i].repeat(grid_size)], 
        axis=1
    ) # 空間座標
    
    # 確率密度を計算
    dens_grid = np.sum(
        [pi_k[k] * multivariate_normal.pdf(x=x_arr, mean=mu_kd[k], cov=sigma2_kdd[k]) for k in range(K_truth)], 
        axis=0
    ).reshape(grid_shape)

    # 計算結果を格納
    dens_lt.append(dens_grid.copy())


# %%

### 観測データの生成 -----

# データ数を指定
N = 100

# 乱数生成器を作成
rng = np.random.default_rng(seed=1010)

# 真のクラスタを生成
c_truth_nk = rng.multinomial(n=1, pvals=pi_k, size=N)

# 真のクラスタ番号を抽出
_, c_truth_n = np.where(c_truth_nk == 1)
print(c_truth_n[:5])

# データを生成
x_nd = np.array(
    [rng.multivariate_normal(mean=mu_kd[k], cov=sigma2_kdd[k], size=1).flatten() for k in c_truth_n]
)
print(x_nd[:5])


# %%

# クラスタリング ----------------------------------------------------------------

### クラスタの推定 -----

## clustering.pyを参照

# 初期化手法を指定
init_type = 'random_samples'

# クラスタの初期値を指定
K_init = 3

# クラスタを推定
trace_res_dic = k_means_method(
    x_nd, K=K_init, 
    lower_clust_num=10, init_centroid=init_type, 
    rng=rng
)


# %%

### 推移の可視化 -----

# 試行回数を取得
iter_num = max(trace_res_dic.keys()) + 1
print(iter_num)

# 目的関数を計算
trace_c_lt = [
    trace_res_dic[iter_cnt][0] for iter_cnt in range(iter_num)
]
trace_z_lt = [
    trace_res_dic[iter_cnt][1] for iter_cnt in range(iter_num)
]
trace_J_lt = [
    np.mean(
        np.sum((x_nd - trace_z_lt[iter_cnt][trace_c_lt[iter_cnt]])**2, axis=1)
    ) for iter_cnt in range(iter_num)
]


# 軸の範囲を設定
u = 5.0
J_min = 0.0
J_max = np.max(trace_J_lt)
J_max = np.ceil(J_max /u)*u  # u単位で切り上げ
print(J_min, J_max)

# 配色の範囲を設定
u = 0.001
dens_min = 0.0
dens_max = np.max(dens_lt)
dens_max = np.ceil(dens_max /u)*u  # u単位で切り上げ
print(dens_min, dens_max)

# 等高線の位置を指定
dens_vals = np.linspace(start=dens_min, stop=dens_max, num=11) # 線の数を指定
print(dens_vals[:5])

# %%

# フレーム数を設定
frame_num = iter_num

# カラーマップを作成
cmap   = plt.get_cmap('gist_rainbow') # カラーマップを指定
colors = cmap(np.arange(K_init) / K_init) # 色データを作成

# グラフオブジェクトを初期化
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('k-means method', fontsize=20)

# 図を分割
gs = fig.add_gridspec(
    nrows=2, ncols=1, height_ratios=[2, 1]
)
ax1  = fig.add_subplot(gs[0], projection='3d')
ax2  = fig.add_subplot(gs[1])
axes = [ax1, ax2]

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    # 値を取得
    iter_cnt  = frame_i # 試行番号
    c_n, z_kd = trace_res_dic[iter_cnt] # クラスタデータ
    K = len(z_kd) # クラスタ数

    ## 推定クラスタの作図

    # クラスタの割当数を集計
    N_k = np.array(
        [np.sum(c_n == k) for k in range(K)]
    ) # 度数

    # 目的関数を計算
    J = np.mean(
        np.sum((x_nd - z_kd[c_n])**2, axis=1) # ノルムの2乗
    ) # ノルムの2乗平均

    # ラベルを作成
    param_lbl  = f'iteration: {iter_cnt}\n'
    param_lbl += f'$N = {N}, K = {len(np.unique(c_n))}, J = {J:.3f}$'

    # 受け皿を初期化
    cent_lt = []
    obs_lt  = []

    # 推定クラスタを描画
    ax = axes[0]
    for i in range(len(x_2_vec)):
        ax.contour(
            x_0_grid, x_1_grid, dens_lt[i], offset=x_2_vec[i], 
            cmap='viridis', vmin=dens_min, vmax=dens_max, levels=dens_vals, alpha=0.5, 
            linewidths=0.33, linestyles='--', 
            zorder=10
        ) # 生成分布
    for k in range(K):
        # クラスタのラベルを作成
        z_str     = ', '.join([f'{z:.1f}' for z in z_kd[k]])
        cent_lbl  = f'$k = {k+1}, z_k = ({z_str})$'
        clust_lbl = f'$k = {k+1}, N_k = {N_k[k]}$'
        
        clust_idx, = np.where(c_n == k) # クラスタの割当インデックス
        ax.plot(
            [[z_kd[k, 0], [x_0_min, z_kd[k, 0], z_kd[k, 0]][d], np.nan] for d in range(D)], 
            [[z_kd[k, 1], [z_kd[k, 1], x_1_max, z_kd[k, 1]][d], np.nan] for d in range(D)], 
            [[z_kd[k, 2], [z_kd[k, 2], z_kd[k, 2], x_2_min][d], np.nan] for d in range(D)], 
            color=colors[k], linestyle='--', 
            zorder=0
        ) # xyz座標の指示線
        ax.plot(
            [[z_kd[k, 0], x_nd[n, 0], np.nan] for n in clust_idx], 
            [[z_kd[k, 1], x_nd[n, 1], np.nan] for n in clust_idx], 
            [[z_kd[k, 2], x_nd[n, 2], np.nan] for n in clust_idx], 
            color=colors[k], linewidth=1.0, linestyle=':', 
            zorder=20
        ) # クラスタの対応線
        cent_sc = ax.scatter(
            xs=z_kd[k, 0], ys=z_kd[k, 1], zs=z_kd[k, 2], 
            facecolor='white', edgecolor=colors[k], s=150, marker='s', 
            label=cent_lbl, 
            zorder=21
        ) # 代表値
        obs_sc = ax.scatter(
            xs=x_nd[clust_idx, 0], ys=x_nd[clust_idx, 1],  zs=x_nd[clust_idx, 2], 
            color=colors[k], s=50, 
            label=clust_lbl, 
            zorder=31
        ) # 観測値

        # 描画オブジェクトを保存
        cent_lt.append(cent_sc)
        obs_lt.append(obs_sc)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_title(param_lbl, loc='left')
    ax.set_xlim(xmin=x_0_min, xmax=x_0_max)
    ax.set_ylim(ymin=x_1_min, ymax=x_1_max)
    ax.set_zlim(zmin=x_2_min, zmax=x_2_max)
    #ax.view_init(elev=30.0, azim=-60.0) # 表示角度

    # 凡例を装飾
    legend1 = ax.legend(
        handles=cent_lt, 
        title='parameters', 
        bbox_to_anchor=(0.0, 1.0), loc='upper right', 
        fontsize=10
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=obs_lt, 
        title='observation data', 
        bbox_to_anchor=(0.0, 0.5), loc='upper right', 
        fontsize=10
    )

    ## 目的関数の推移の作図

    # 目的関数の推移を描画
    ax = axes[1]
    ax.plot(
        np.arange(iter_cnt+1), trace_J_lt[:(iter_cnt+1)]
    ) # 目的関数の推移
    ax.scatter(
        x=iter_cnt, y=trace_J_lt[iter_cnt], 
        s=50
    ) # 目的関数の現在値

    ax.set_xlabel('iteration')
    ax.set_ylabel('$J = \\frac{1}{n} \\sum_{i=1}^n \\|x_i - z_{c_i}\\|^2$')
    ax.grid()
    ax.set_xlim(xmin=0, xmax=iter_num-1)
    ax.set_ylim(ymin=J_min, ymax=J_max)

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=250
)

# 動画を書出
anim.save(
    filename=dir_path+f'3d_clustering_{init_type}_init_K_{K_init}.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


