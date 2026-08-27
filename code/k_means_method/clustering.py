
# k-means法 --------------------------------------------------------------------

# ch 4.3

# クラスタリング
# アルゴリズムの実装


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np


# %%

# クラスタの推定用 ---------------------------------------------------------------

# k-means法を実装
def k_means_method(
    X, K=10, 
    threshold=0.001, max_iter=100, 
    lower_clust_num=0, init_centroid='random_samples', 
    rng=None
):
    # データ数・次元数を取得
    x_nd = X
    N, D = x_nd.shape

    # 乱数生成器を作成
    if rng is None:
        rng = np.random.default_rng()

    # カウントを初期化
    iter_cnt = 0

    # k-meansによるクラスタリング
    while iter_cnt < max_iter: # 無限ループの回避用
    
        # クラスタの代表値を計算
        if iter_cnt > 0: # 初回を除く
            z_kd = np.array(
                [np.mean(x_nd[c_n == k], axis=0) for k in range(K) if N_k[k] >= lower_clust_num]
            ) # 平均値
        elif init_centroid == 'random_samples': # 観測値を代表値とする場合
            rand_idx = rng.choice(a=N, size=K, replace=False) # 代表値を割当
            z_kd     = x_nd[rand_idx].copy() # 観測値
        elif init_centroid == 'random_uniform': # 観測値を代表値とする場合
            z_kd = np.stack(
                [rng.uniform(low=x_nd[:, d].min(), high=x_nd[:, d].max(), size=K) for d in range(D)], 
                axis=1
            ) # 一様乱数
        elif init_centroid == 'random_clusters': # クラスタをランダムに割り当てる場合
            c_n  = rng.choice(a=np.arange(K), size=N, replace=True) # クラスタを割当
            z_kd = np.array(
                [np.mean(x_nd[c_n == k], axis=0) for k in range(K)]
            ) # 平均値
        else:
            raise ValueError(
                f'Unknown centroid initialization method: {init_centroid!r}.\n'
                "Expected one of: 'random_samples', 'random_uniform', 'random_clusters'"
            )

        # クラスタ数を更新
        K = len(z_kd)

        # クラスタを割当
        c_n = np.argmin(
            [np.linalg.norm(x_nd - z_kd[k], axis=1) for k in range(K)], # ノルム
            axis=0
        ) # 最小ノルムのインデックス
    
        # クラスタの割当数を集計
        N_k = np.array(
            [np.sum(c_n == k) for k in range(K)]
        ) # 度数 
    
        # 目的関数を計算
        J = np.mean(
            np.sum((x_nd - z_kd[c_n])**2, axis=1) # ノルムの2乗
        ) # ノルムの2乗平均

        # 試行終了を判定
        if iter_cnt > 0: # 初回を除く
            delta_J = abs(J - old_J) # 更新量
            if delta_J < threshold: # 変化がない場合
                break

        # 更新値を記録
        old_J = J

        # 試行回数をカウント
        iter_cnt += 1

    # クラスタ番号を出力
    return c_n


# %%

# 推定推移の可視化用 -------------------------------------------------------------

# k-means法を実装
def k_means_method(
    X, K=10, 
    threshold=0.001, max_iter=100, 
    lower_clust_num=0, init_centroid='random_samples', 
    rng=None
):
    # データ数・次元数を取得
    x_nd = X
    N, D = x_nd.shape

    # 乱数生成器を作成
    if rng is None:
        rng = np.random.default_rng()

    # 受け皿を初期化
    trace_res_dic = {}

    # カウントを初期化
    iter_cnt = 0

    # k-meansによるクラスタリング
    while iter_cnt < max_iter: # 無限ループの回避用
    
        # クラスタの代表値を計算
        if iter_cnt > 0: # 初回を除く
            z_kd = np.array(
                [np.mean(x_nd[c_n == k], axis=0) if N_k[k] >= lower_clust_num else np.tile(np.nan, reps=D) for k in range(K)]
            ) # 平均値
        elif init_centroid == 'random_samples': # 観測値を代表値とする場合
            rand_idx = rng.choice(a=N, size=K, replace=False) # 代表値を割当
            z_kd     = x_nd[rand_idx].copy() # 観測値
        elif init_centroid == 'random_uniform': # 観測値を代表値とする場合
            z_kd = np.stack(
                [rng.uniform(low=x_nd[:, d].min(), high=x_nd[:, d].max(), size=K) for d in range(D)], 
                axis=1
            ) # 一様乱数
        elif init_centroid == 'random_clusters': # クラスタをランダムに割り当てる場合
            c_n  = rng.choice(a=np.arange(K), size=N, replace=True) # クラスタを割当
            z_kd = np.array(
                [np.mean(x_nd[c_n == k], axis=0) for k in range(K)]
            ) # 平均値
        else:
            raise ValueError(
                f'Unknown centroid initialization method: {init_centroid!r}.\n'
                "Expected one of: 'random_samples', 'random_uniform', 'random_clusters'"
            )

        # クラスタ数を更新
        K = len(z_kd)

        # クラスタを割当
        c_n = np.nanargmin(
            [np.linalg.norm(x_nd - z_kd[k], axis=1) for k in range(K)], # ノルム
            axis=0
        ) # 最小ノルムのインデックス
    
        # クラスタの割当数を集計
        N_k = np.array(
            [np.sum(c_n == k) for k in range(K)]
        ) # 度数 
    
        # 目的関数を計算
        J = np.nanmean(
            np.nansum((x_nd - z_kd[c_n])**2, axis=1) # ノルムの2乗
        ) # ノルムの2乗平均
    
        # 試行終了を判定
        if iter_cnt > 0: # 初回を除く
            delta_J = abs(J - old_J) # 更新量
            if delta_J < threshold: # 変化がない場合
                break

        # 更新値を記録
        old_J = J
        trace_res_dic[iter_cnt] = (c_n.copy(), z_kd.copy())

        # 試行回数をカウント
        iter_cnt += 1

    # クラスタ番号を出力
    return trace_res_dic


# %%


