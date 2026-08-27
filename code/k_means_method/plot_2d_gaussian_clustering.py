
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

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

# トイデータの作成 ---------------------------------------------------------------

### 生成分布の設定 -----

# 真のクラスタ数を指定
K_truth = 3

# K個の平均ベクトルを指定
mu_kd = np.array(
    [[0.0, 5.0], 
     [5.0, -10.0], 
     [-10.0, -5.0]]
)

# K個の分散共分散行列を指定
sigma2_kdd = np.array(
    [[[36.0, 10.0], [10.0, 25.0]], 
     [[9.0, -1.3], [-1.3, 16.0]], 
     [[25.0, -3.2], [-3.2, 16.0]]]
)

# 混合比率を指定
pi_k = np.array([0.45, 0.25, 0.3])


# %%

### 生成分布の計算 -----

# x軸の範囲を設定
u       = 1.0
sgm_num = 3.0
x_1_min, x_2_min = np.min(
    [mu_kd[k] - sgm_num * np.sqrt(np.diag(sigma2_kdd[k])) for k in range(K_truth)], 
    axis=0
)
x_1_max, x_2_max = np.max(
    [mu_kd[k] + sgm_num * np.sqrt(np.diag(sigma2_kdd[k])) for k in range(K_truth)], 
    axis=0
)
x_1_min, x_2_min = np.floor(np.array([x_1_min, x_2_min]) /u)*u # u単位で切り下げ
x_1_max, x_2_max = np.ceil(np.array([x_1_max, x_2_max]) /u)*u  # u単位で切り上げ
print(x_1_min, x_1_max)
print(x_2_min, x_2_max)

# x軸の値を作成
x_1_vec = np.linspace(start=x_1_min, stop=x_1_max, num=251)
x_2_vec = np.linspace(start=x_2_min, stop=x_2_max, num=251)

# 格子点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_vec, x_2_vec)

# 座標を作成
x_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)

# %%

# 確率密度を計算
dens_grid = np.sum(
    [pi_k[k] * multivariate_normal.pdf(x=x_arr, mean=mu_kd[k], cov=sigma2_kdd[k]) for k in range(K_truth)], 
    axis=0
).reshape(x_1_grid.shape)


# %%

### 観測データの生成 -----

# データ数を指定
N = 100

# 乱数生成器を作成
rng = np.random.default_rng(seed=102)

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

# %%

# フレーム数を設定
frame_num = iter_num

# カラーマップを作成
cmap   = plt.get_cmap('gist_rainbow') # カラーマップを指定
colors = cmap(np.arange(K_init) / K_init) # 色データを作成

# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=2, ncols=1, height_ratios=[2, 1], 
    figsize=(9, 9), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('k-means method', fontsize=20)

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
    ax.contour(
        x_1_grid, x_2_grid, dens_grid, 
        linewidths=1.0, linestyles='--', 
        zorder=10
    ) # 生成分布
    for k in range(K):
        # クラスタのラベルを作成
        z_str     = ', '.join([f'{z:.1f}' for z in z_kd[k]])
        cent_lbl  = f'$k = {k+1}, z_k = ({z_str})$'
        clust_lbl = f'$k = {k+1}, N_k = {N_k[k]}$'

        clust_idx, = np.where(c_n == k) # クラスタの割当インデックス
        ax.plot(
            [z_kd[k, 0].repeat(N_k[k]), x_nd[clust_idx, 0]], 
            [z_kd[k, 1].repeat(N_k[k]), x_nd[clust_idx, 1]], 
            color=colors[k], linewidth=1.0, linestyle=':', 
            zorder=20
        ) # クラスタの対応線
        cent_sc = ax.scatter(
            x=z_kd[k, 0], y=z_kd[k, 1], 
            facecolor='white', edgecolor=colors[k], s=150, marker='s', 
            label=cent_lbl, 
            zorder=21
        ) # 代表値
        obs_sc = ax.scatter(
            x=x_nd[clust_idx, 0], y=x_nd[clust_idx, 1], 
            color=colors[k], s=50, 
            label=clust_lbl, 
            zorder=31
        ) # 観測値

        # 描画オブジェクトを保存
        cent_lt.append(cent_sc)
        obs_lt.append(obs_sc)

    # 凡例を装飾
    legend1 = ax.legend(
        handles=obs_lt, 
        title='observation data', 
        bbox_to_anchor=(1.0, 1.0), loc='upper left', 
        fontsize=10
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=cent_lt, 
        title='centroids', 
        bbox_to_anchor=(1.0, 0.5), loc='upper left', 
        fontsize=10
    )

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(param_lbl, loc='left')
    ax.grid()
    ax.set_xlim(xmin=x_1_min, xmax=x_1_max)
    ax.set_ylim(ymin=x_2_min, ymax=x_2_max)

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
    filename=dir_path+f'2d_clustering_{init_type}_init_K_{K_init}.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


