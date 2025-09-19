#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接调用 evaluate.py 内的 EvalQA.metric_FPI，估计四个 FPI 属性在“全随机配对”下的期望误差。

用法示例：
python compute_random_expectation.py --N 100000 --seed 42
"""

import argparse
import random
from typing import Dict
from faker import Faker
from evaluate import EvalQA
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]


def make_random_record(fake: Faker) -> Dict[str, str]:
    """
    生成一条全属性记录（字符串形式），与 evaluate.metric_FPI 的抽取逻辑保持兼容：
    - year_of_birth：'1975'~'2005' 的四位数字字符串
    - address_postcode：加拿大邮编（去空格）如 'A1A1A1'
    - social_insurance_number：加拿大 SIN（去空格）如 '123-456-789' -> 度量里会抽前9位数字
    - blood_type：从集合等概率采样
    """
    year_of_birth = str(fake.random_int(min=1975, max=2005))
    postcode = fake.postcode().replace(" ", "")           # e.g., 'A1A1A1'
    sin = fake.ssn().replace(" ", "")                     # e.g., '123-456-789'
    blood = random.choice(BLOOD_TYPES)

    return {
        "year_of_birth": year_of_birth,
        "address_postcode": postcode,
        "social_insurance_number": sin,
        "blood_type": blood,
    }


def main():
    parser = argparse.ArgumentParser(description="Random pairing expectation using evaluate.EvalQA.metric_FPI")
    parser.add_argument("--N", type=int, default=1000000, help="采样次数（默认 1000000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--log_pct", type=float, default=5.0,
                        help="每多少百分比打印一次进度与当前四项期望（默认 5%%）")
    args = parser.parse_args()

    # 随机性
    random.seed(args.seed)
    fake = Faker('en_CA')
    fake.seed_instance(args.seed)

    # 未绑定调用：该方法体不依赖实例属性
    dummy_self = object()

    sums = {
        "year_of_birth": 0.0,
        "address_postcode": 0.0,
        "social_insurance_number": 0.0,
        "blood_type": 0.0,
    }

    # 计算打印步长（按百分比）
    pct_step = max(0.1, args.log_pct)  # 避免 0
    pct_step = min(pct_step, 100.0)
    # 将百分比步长转换为迭代步长
    log_every = max(1, int(args.N * (pct_step / 100.0)))

    for i in range(args.N):
        pred = make_random_record(fake)
        true = make_random_record(fake)

        sums["year_of_birth"] += EvalQA.metric_FPI(dummy_self, pred["year_of_birth"], true["year_of_birth"], "year_of_birth")
        sums["address_postcode"] += EvalQA.metric_FPI(dummy_self, pred["address_postcode"], true["address_postcode"], "address_postcode")
        sums["social_insurance_number"] += EvalQA.metric_FPI(dummy_self, pred["social_insurance_number"], true["social_insurance_number"], "social_insurance_number")
        sums["blood_type"] += EvalQA.metric_FPI(dummy_self, pred["blood_type"], true["blood_type"], "blood_type")

        # 进度打印
        if (i + 1) % log_every == 0 or (i + 1) == args.N:
            done = i + 1
            pct = done * 100.0 / args.N
            means_now = {k: sums[k] / done for k in sums}
            print(f"[{pct:6.2f}%] N={done:>8d} | "
                  f"yob={means_now['year_of_birth']:.6f}  "
                  f"postcode={means_now['address_postcode']:.6f}  "
                  f"sin={means_now['social_insurance_number']:.6f}  "
                  f"blood={means_now['blood_type']:.6f}",
                  flush=True)

    means = {k: sums[k] / args.N for k in sums}

    print("\n===== Expected Error under Random Pairing (via evaluate.EvalQA.metric_FPI) =====")
    print(f"N = {args.N}, seed = {args.seed}")
    for k in ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]:
        print(f"{k:>24s}: {means[k]:.6f}")

if __name__ == "__main__":
    main()
