# -*- coding: utf-8 -*-
"""NumPy 数值计算练习（配套 Project 1）

对应练习册 practice/project1_exercises.pdf：
    8 道基础题 NP-E1 ~ NP-E8
    4 道中等题 NP-M1 ~ NP-M4

运行方式：
    python numpy_homework.py

每题都直接打印结果，函数题保留题目指定的函数名。
关键表达式旁边写了“为什么这样写”，方便讲评时解释 shape。
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def title(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


# ==========================================================================
# NP-E1 数组、形状与切片
# ==========================================================================
def np_e1():
    title("NP-E1 数组、形状与切片")
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])
    A_before = A.copy()        # 留一份原件，最后核对 A 没被改动

    # 1. 三个基本属性：维数、形状（元组）、元素个数
    print("A.ndim =", A.ndim)
    print("A.shape =", A.shape, " A.size =", A.size)

    # 2. 第 2 行（下标 1）是一维；最后 2 列是二维（保留行维）
    row = A[1]
    last2 = A[:, -2:]          # 冒号：所有行；-2: 最后两列
    print("A[1]        =", row, " shape =", row.shape)
    print("A[:, -2:]   =", last2.tolist(), " shape =", last2.shape)

    # 3. 先复制再改，原数组不受影响（切片是视图，copy 才是独立内存）
    B = A.copy()
    B[B < 0] = 0.0             # 布尔索引：把满足条件的元素一次改掉
    print("B =", B.tolist())
    print("A 未被修改：", np.array_equal(A, A_before))


# ==========================================================================
# NP-E2 逐元素乘法、点乘与长度
# ==========================================================================
def np_e2():
    title("NP-E2 逐元素乘法、点乘与长度")
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # * 是逐元素，形状不变；@ 是点乘，两个一维数组得到一个标量
    elem = a * b
    dot = a @ b
    print("a * b =", elem, " shape =", elem.shape)
    print("a @ b =", dot, " 类型 =", type(dot).__name__)

    # 长度的三种写法：平方、求和、开方
    len_a = np.sqrt(np.sum(a * a))
    len_b = np.sqrt((b ** 2).sum())
    print("|a| =", len_a, " |b| =", len_b)

    # 用 np.linalg.norm 核对（默认算二范数）
    print("np.linalg.norm 核对：", np.allclose(len_a, np.linalg.norm(a)),
          np.allclose(len_b, np.linalg.norm(b)))


# ==========================================================================
# NP-E3 两个向量的夹角
# ==========================================================================
def angle_deg(a, b):
    """返回两个非零一维向量的夹角（度）。任一为零向量时抛 ValueError。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("零向量没有方向，无法定义夹角")
    cos = (a @ b) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)      # 浮点误差可能让 |cos| 略大于 1
    return float(np.degrees(np.arccos(cos)))


def np_e3():
    title("NP-E3 两个向量的夹角")
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    print("angle_deg(a, b) = %.4f°" % angle_deg(a, b))   # 应为 45°

    try:
        angle_deg(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    except ValueError as exc:
        print("零向量测试 ->", exc)


# ==========================================================================
# NP-E4 两个点的距离
# ==========================================================================
def np_e4():
    title("NP-E4 两个点的距离")
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    diff = p - q
    euclid = np.sqrt(np.sum(diff ** 2))
    manhattan = np.sum(np.abs(diff))
    print("p - q =", diff)
    print("欧氏距离 = %.4f" % euclid)
    print("曼哈顿距离 = %.4f" % manhattan)
    print("交叉核对 |diff| 的范数 =", np.linalg.norm(diff))


# ==========================================================================
# NP-E5 把向量变为单位长度
# ==========================================================================
def unit_vector(a):
    """返回与 a 方向相同、长度为 1 的新数组（不修改 a）。"""
    a = np.asarray(a, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError("零向量无法单位化")
    return a / norm


def np_e5():
    title("NP-E5 把向量变为单位长度")
    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print("u =", u)
    print("长度 =", np.linalg.norm(u), " 与原数组分离 =", u is not a)
    print("np.allclose(|u|, 1) ->", np.allclose(np.linalg.norm(u), 1.0))
    print("原向量保持不变：", a)


# ==========================================================================
# NP-E6 沿不同 axis 汇总
# ==========================================================================
def np_e6():
    title("NP-E6 沿不同 axis 汇总")
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])

    s0 = A.sum(axis=0)      # 跨行相加 -> 每列一个值，shape (3,)
    m1 = A.mean(axis=1)     # 跨列求平均 -> 每行一个值，shape (2,)
    mx = A.max()            # 不写 axis -> 整个数组一个标量
    print("A.sum(axis=0)  =", s0, " shape =", s0.shape, "（每列之和）")
    print("A.mean(axis=1) =", m1, " shape =", m1.shape, "（每行均值）")
    print("A.max()        =", mx, "（整体最大值）")

    # 直观解释：axis 就是“被压缩掉的那一维”
    print("核对：s0 等于逐列相加 =", np.allclose(s0, [A[0, j] + A[1, j] for j in range(3)]))


# ==========================================================================
# NP-E7 一次广播，处理所有行
# ==========================================================================
def np_e7():
    title("NP-E7 一次广播，处理所有行")
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    # X 是 (2,3)，scale/bias 是 (3,)：NumPy 自动把 (3,) 当成 (1,3) 铺到两行
    Y = X * scale + bias
    print("Y =", Y.tolist(), " shape =", Y.shape)
    print("逐行核对 =", np.allclose(Y[1], X[1] * scale + bias))


# ==========================================================================
# NP-E8 把两张图像组成一个 batch
# ==========================================================================
def np_e8():
    title("NP-E8 把两张图像组成一个 batch")
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])   # (C=3, H=2, W=2)
    b = a + 0.10

    batch = np.stack([a, b], axis=0)     # 新插入的第 0 轴就是“图像编号”
    print("batch.shape =", batch.shape)  # (2, 3, 2, 2)
    blue = batch[:, 2, :, :]             # C=2 是蓝色通道
    print("蓝色通道 shape =", blue.shape)  # (2, 2, 2)
    hwc = batch.transpose(0, 2, 3, 1)    # (B, C, H, W) -> (B, H, W, C)
    print("(B,H,W,C) shape =", hwc.shape)

    # 单张图也能用 np.stack 组 batch，leading 维为 1
    single = np.stack([a], axis=0)
    print("单张组 batch shape =", single.shape)


# ==========================================================================
# NP-M1 批量计算夹角
# ==========================================================================
def angles_to(query, X):
    """返回 X 每一行与 query 的夹角（度），shape 为 (n,)。

    零向量的行返回 nan；query 为零向量时抛 ValueError。
    不使用遍历 X 各行的循环。
    """
    query = np.asarray(query, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        raise ValueError("query 不能是零向量")

    row_norms = np.sqrt((X ** 2).sum(axis=1))       # (n,)
    dots = X @ query                                # (n,) 一次矩阵乘法
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = dots / (row_norms * q_norm)           # 零行会得到 inf/nan
    cos[row_norms == 0] = np.nan                    # 显式保证零行为 nan
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def np_m1():
    title("NP-M1 批量计算夹角")
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])

    angles = angles_to(query, X)
    print("angles =", angles, " shape =", angles.shape)
    print("零向量行 ->", angles[3], "（nan 表示无效）")
    print("逐行核对第 0 行 = %.4f" % angle_deg(query, X[0]))


# ==========================================================================
# NP-M2 用余弦相似度做 Top-k 检索
# ==========================================================================
def top_k_cosine(query, gallery, k):
    """按余弦相似度从高到低返回 (前 k 个行下标, 对应相似度)。

    零向量的 gallery 行不参与排序；零 query 抛 ValueError；
    k 必须是正整数且不超过有效行数。
    """
    query = np.asarray(query, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        raise ValueError("query 不能是零向量")

    norms = np.sqrt((gallery ** 2).sum(axis=1))
    valid = norms > 0
    if not isinstance(k, (int, np.integer)) or k <= 0 or k > int(valid.sum()):
        raise ValueError("k 必须是正整数，且不超过非零行数")

    # 只对非零行算相似度；零行不参与排序
    idx_valid = np.flatnonzero(valid)
    scores = (gallery[idx_valid] @ query) / (norms[idx_valid] * q_norm)
    scores = np.clip(scores, -1.0, 1.0)

    # 稳定排序：分数相同时保持原来的行顺序
    order = np.argsort(-scores, kind="stable")[:k]
    return idx_valid[order], scores[order]


def np_m2():
    title("NP-M2 用余弦相似度做 Top-k 检索")
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])

    top_idx, top_score = top_k_cosine(query, gallery, k=3)
    print("top_idx =", top_idx, " scores =", top_score)
    print("选中 ID =", ids[top_idx].tolist())

    for bad_k in (0, 6):
        try:
            top_k_cosine(query, gallery, bad_k)
        except ValueError as exc:
            print(f"k={bad_k} ->", exc)


# ==========================================================================
# NP-M3 所有点对之间的距离矩阵
# ==========================================================================
def np_m3():
    title("NP-M3 所有点对之间的距离矩阵")
    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])

    # A[:, None, :] 是 (3,1,2)，B[None, :, :] 是 (1,2,2)
    # 相减时广播成 (3,2,2)：D[i,j] 就是 A[i]-B[j]
    diff = A[:, None, :] - B[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    print("diff.shape =", diff.shape, "（广播中间结果）")
    print("D =\n", D, "\nD.shape =", D.shape)

    nearest = D.argmin(axis=1)      # 每行最小值的列下标
    print("A 每行最近的 B 行下标 =", nearest)


# ==========================================================================
# NP-M4 用矩阵运算完成一次前向计算
# ==========================================================================
def np_m4():
    title("NP-M4 用矩阵运算完成一次前向计算")
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])
    params_before = [w.copy() for w in (X, W1, b1, W2, b2)]

    # 权重按 [输出特征, 输入特征] 存储，所以右乘 W.T
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)          # ReLU
    Y = H @ W2.T + b2
    print("X.shape =", X.shape)
    print("Z =\n", Z, "\nZ.shape =", Z.shape)
    print("H =\n", H, "\nH.shape =", H.shape)
    print("Y =\n", Y, "\nY.shape =", Y.shape)
    print("参数未被修改：", all(np.array_equal(a, b) for a, b in zip(params_before, (X, W1, b1, W2, b2))))


if __name__ == "__main__":
    np_e1()
    np_e2()
    np_e3()
    np_e4()
    np_e5()
    np_e6()
    np_e7()
    np_e8()
    np_m1()
    np_m2()
    np_m3()
    np_m4()
    print("\n全部练习运行完毕。")

