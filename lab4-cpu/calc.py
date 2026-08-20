import math

def compute_score_params(t1=340, s1=100, t2=500, s2=60):
    """
    根据两个基准点 (t1, s1) 和 (t2, s2)，
    求解 y = a * x^b 中的 a, b
    """
    # b = ln(s1/s2) / ln(t1/t2)
    b = math.log(s1 / s2) / math.log(t1 / t2)
    # a = s1 / t1^b
    a = s1 / (t1 ** b)
    return a, b


def score(program_time, a, b, max_score=100.0, min_score=0.0):
    """
    根据运行时间计算得分：
    - program_time <= 0 时没有意义，直接返回 max_score（或按需处理）
    - y = a * t^b，并做上下限截断（clamp）
    """
    if program_time <= 0:
        return max_score

    raw_score = a * (program_time ** b)

    # 封顶/保底处理
    raw_score = min(raw_score, max_score)
    raw_score = max(raw_score, min_score)

    return raw_score


if __name__ == "__main__":
    # 计算 a, b（基于 (340,100) 和 (500,60) 两个点）
    a, b = compute_score_params(340, 100, 500, 60)
    print(f"a = {a:.6f}, b = {b:.6f}")

    # 验证两个基准点
    print("t=340 ->", score(340, a, b))  # 应为 100
    print("t=500 ->", score(500, a, b))  # 应为 60

    # 测试一批运行时间
    test_times = float(input())
    print("\ntime(s)\tscore")
    for t in test_times:
        s = score(t, a, b)
        print(f"{t}\t{s:.2f}")