import sys
import math

def compute_score_params(t1=340, s1=100, t2=500, s2=60):
    b = math.log(s1 / s2) / math.log(t1 / t2)
    a = s1 / (t1 ** b)
    return a, b


def score(program_time, a, b, max_score=100.0, min_score=0.0):
    if program_time <= 0:
        return max_score
    raw_score = a * (program_time ** b)
    raw_score = min(raw_score, max_score)
    raw_score = max(raw_score, min_score)
    return raw_score


if __name__ == "__main__":
    a, b = compute_score_params(340, 100, 500, 60)

    if len(sys.argv) > 1:
        # 脚本传参调用：python3 calc.py <program_time>
        t = float(sys.argv[1])
    else:
        # 交互式调用：手动输入一个时间
        t = float(input("program time (s): "))

    s = score(t, a, b)
    print(f"program_time={t}\tscore={s:.2f}")