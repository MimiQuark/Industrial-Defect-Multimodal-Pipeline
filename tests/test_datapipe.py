#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datapipe 数据管线测试（无 pytest 依赖，直接 python tests/test_datapipe.py 运行）

覆盖：
1. mask 标注提取正确性（已知几何形状 → 断言面积/位置/bbox）
2. 九宫格方位映射
3. dHash 感知哈希与去重逻辑
4. JSONL 格式校验逻辑（含一致性检查）
5. 模板改写可复现性
6. 管线 B 启发式打分（劣化图应得低分）
7. 端到端 smoke test（临时目录微型 demo 数据集 → JSONL）
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.annotation import (DEFECT_TYPE_ZH, MVTEC_DEFECT_TYPES,    # noqa: E402
                                 extract_annotation_from_mask,
                                 position_zh_from_centroid, scan_mvtec_dataset)
from datapipe.make_demo_dataset import build_demo_dataset               # noqa: E402
from datapipe.pipeline_b import heuristic_score, image_stats, reference_distribution  # noqa: E402
from datapipe.quality import (dedup_by_phash, dhash, hamming_distance,  # noqa: E402
                              validate_record, write_version)
from datapipe.rewrite import rewrite_annotations, template_rewrite      # noqa: E402
import random                                                           # noqa: E402


def test_mask_annotation():
    """已知几何形状 mask → 面积/位置/bbox 断言"""
    print("[1] mask 标注提取正确性...")
    with tempfile.TemporaryDirectory() as td:
        mask_path = Path(td) / "m.png"
        # 256x256 上画矩形 [50,60]~[110,140]（PIL ellipse/rectangle 含端点）
        img = Image.new("L", (256, 256), 0)
        ImageDraw.Draw(img).rectangle([50, 60, 110, 140], fill=255)
        img.save(mask_path)
        info = extract_annotation_from_mask(mask_path)
        expected_area = 61 * 81 / (256 * 256)
        assert abs(info["area_ratio"] - expected_area) < 1e-9, \
            f"面积不符: {info['area_ratio']} != {expected_area}"
        assert info["bbox"] == [50, 60, 110, 140], f"bbox 不符: {info['bbox']}"
        # 质心 (80, 100) → 归一化 (0.3125, 0.3906) → 左列中行 → 左侧区域
        assert abs(info["centroid"][0] - 80 / 256) < 0.01
        assert abs(info["centroid"][1] - 100 / 256) < 0.01
        assert info["position_zh"] == "左侧区域", info["position_zh"]
        # 空 mask：缺陷样本的空 mask 属于数据错误，必须显式失败
        empty = Image.new("L", (256, 256), 0)
        empty_path = Path(td) / "empty.png"
        empty.save(empty_path)
        try:
            extract_annotation_from_mask(empty_path)
        except ValueError as e:
            assert "前景" in str(e), e
        else:
            raise AssertionError("空 mask 应抛 ValueError")
        # 尺寸不一致必须显式失败
        small = Image.new("L", (128, 128), 255)
        small_path = Path(td) / "small.png"
        small.save(small_path)
        try:
            extract_annotation_from_mask(small_path, (256, 256))
        except ValueError as e:
            assert "尺寸" in str(e), e
        else:
            raise AssertionError("mask/图像尺寸不一致应抛 ValueError")
    print("   ✓ 面积/bbox/质心/方位/异常输入断言通过")


def test_defect_type_mapping():
    """MVTec AD 全部缺陷目录名都有中文映射（防止 VLM 一致性校验静默丢弃）"""
    print("[1b] 缺陷类型映射完整性...")
    missing = sorted(MVTEC_DEFECT_TYPES - set(DEFECT_TYPE_ZH))
    assert not missing, f"缺少中文映射: {missing}"
    for en in MVTEC_DEFECT_TYPES:
        zh = DEFECT_TYPE_ZH[en]
        assert zh and zh != en, f"{en} 未映射为中文: {zh!r}"
    assert len(DEFECT_TYPE_ZH) >= len(MVTEC_DEFECT_TYPES)
    print(f"   ✓ {len(MVTEC_DEFECT_TYPES)} 个 MVTec 缺陷目录名全部有中文映射")


def test_position_grid():
    """九宫格方位映射"""
    print("[2] 九宫格方位映射...")
    assert position_zh_from_centroid(0.9, 0.1) == "右上区域"
    assert position_zh_from_centroid(0.1, 0.9) == "左下区域"
    assert position_zh_from_centroid(0.5, 0.5) == "中央区域"
    assert position_zh_from_centroid(0.5, 0.1) == "上部区域"
    assert position_zh_from_centroid(0.1, 0.5) == "左侧区域"
    assert position_zh_from_centroid(0.5, 0.9) == "下部区域"
    print("   ✓ 6 个方位断言通过")


def test_dhash_dedup():
    """dHash：相同图距离 0，近似图距离小，去重剔除复制图"""
    print("[3] dHash 与去重逻辑...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.RandomState(0)
        base = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        base.save(td / "a.png")
        shutil.copyfile(td / "a.png", td / "a_copy.png")
        # 近似图：加微小噪声
        arr = np.asarray(base).astype(np.int16) + rng.randint(-2, 3, (64, 64, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(td / "a_noisy.png")
        other = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        other.save(td / "b.png")

        h_a, h_copy = dhash(td / "a.png"), dhash(td / "a_copy.png")
        assert hamming_distance(h_a, h_copy) == 0, "相同图距离应为 0"
        d_noisy = hamming_distance(h_a, dhash(td / "a_noisy.png"))
        d_other = hamming_distance(h_a, dhash(td / "b.png"))
        assert d_noisy < d_other, f"近似图距离({d_noisy})应小于无关图({d_other})"

        recs = [{"image": n, "instruction": "i", "response": "r"}
                for n in ("a.png", "a_copy.png", "b.png")]
        kept, dropped = dedup_by_phash(recs, td, hamming_threshold=5)
        assert len(kept) == 2 and len(dropped) == 1
        assert dropped[0]["image"] == "a_copy.png"
        assert dropped[0]["dup_of"] == "a.png"
    print("   ✓ 哈希距离与去重断言通过")


def test_validation():
    """JSONL 校验：缺字段/空字段/一致性失败均能被检出"""
    print("[4] JSONL 格式校验逻辑...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        Image.new("RGB", (8, 8)).save(td / "x.png")
        ok = {"image": "x.png", "instruction": "检查缺陷", "response": "存在划痕缺陷",
              "meta": {"annotation": {"defect_type": "scratch"}}}
        assert validate_record(ok, td) == []
        # 缺字段
        assert any("instruction" in e for e in validate_record(
            {"image": "x.png", "response": "r"}, td))
        # 图像不存在
        assert any("不存在" in e for e in validate_record(
            {**ok, "image": "nope.png"}, td))
        # response 为空
        assert any("response 为空" in e for e in validate_record(
            {**ok, "response": "  "}, td))
        # 一致性：response 未包含缺陷类型
        errs = validate_record({**ok, "response": "一切正常"}, td)
        assert any("一致性" in e for e in errs), errs
        # 一致性：good 样本 response 未表明无缺陷
        bad_good = {"image": "x.png", "instruction": "i",
                    "response": "存在明显划痕",
                    "meta": {"annotation": {"defect_type": "good"}}}
        assert any("一致性" in e for e in validate_record(bad_good, td))
        # 关闭一致性检查则通过
        assert validate_record(bad_good, td, consistency_check=False) == []
    print("   ✓ 6 组校验断言通过")


def test_template_reproducibility():
    """模板降级：同种子输出完全一致"""
    print("[5] 模板改写可复现性...")
    anns1 = None
    with tempfile.TemporaryDirectory() as td:
        root = build_demo_dataset(Path(td) / "ds", seed=7, categories=["bottle"],
                                  n_train_good=1, n_test_good=2,
                                  n_per_defect=1, with_duplicate=False)
        anns = scan_mvtec_dataset(Path(root["root"]))
        r1, m1 = rewrite_annotations(anns, Path(root["root"]),
                                     {"enable_vlm": False}, seed=99)
        r2, m2 = rewrite_annotations(anns, Path(root["root"]),
                                     {"enable_vlm": False}, seed=99)
        assert m1 == m2 == "template_fallback"
        assert [r["response"] for r in r1] == [r["response"] for r in r2], \
            "同种子输出必须一致"
        assert all(r["meta"]["rewrite_mode"] == "template_fallback" for r in r1)
    print("   ✓ 同种子两次运行输出一致")


def test_partial_fallback_accounting():
    """部分 VLM 失败时应标为 vlm_partial_fallback，而非整体 vlm"""
    print("[5b] VLM 部分回退记账...")
    import datapipe.rewrite as rewrite_mod
    with tempfile.TemporaryDirectory() as td:
        root = build_demo_dataset(Path(td) / "ds", seed=3, categories=["bottle"],
                                  n_train_good=1, n_test_good=2,
                                  n_per_defect=2, with_duplicate=False)
        anns = scan_mvtec_dataset(Path(root["root"]))
        orig = rewrite_mod.vlm_rewrite

        def fake_vlm(ann, image_path, vlm_cfg, api_key, rng):
            if not ann.has_defect:  # 正常样本模拟调用失败
                raise RuntimeError("simulated vlm failure")
            return {"instruction": "请分析该图像",
                    "response": f"该图像存在{ann.defect_type_zh}缺陷，判定不合格。"}

        rewrite_mod.vlm_rewrite = fake_vlm
        old_key = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        try:
            recs, mode = rewrite_annotations(
                anns, Path(root["root"]),
                {"enable_vlm": True, "concurrency": 1,
                 "base_url": "http://example.invalid", "model": "fake"},
                seed=5)
        finally:
            rewrite_mod.vlm_rewrite = orig
            if old_key is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = old_key
        assert mode == "vlm_partial_fallback", mode
        by_mode = {}
        for r in recs:
            by_mode[r["meta"]["rewrite_mode"]] = \
                by_mode.get(r["meta"]["rewrite_mode"], 0) + 1
        assert by_mode.get("vlm", 0) == 4, by_mode
        assert by_mode.get("template_fallback", 0) == 2, by_mode
    print("   ✓ 4 条 VLM + 2 条回退，整体标记为 vlm_partial_fallback")


def test_heuristic_scoring():
    """管线 B 启发式打分：过曝/塌陷图应得低分"""
    print("[6] 启发式降级打分...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.RandomState(1)
        good_dir = td / "ref" / "cat" / "test" / "good"
        good_dir.mkdir(parents=True)
        base_arr = rng.randint(100, 160, (64, 64), dtype=np.uint8)
        for i in range(4):
            Image.fromarray(base_arr + rng.randint(-3, 4, (64, 64)).astype(np.uint8)
                            ).save(good_dir / f"{i}.png")
        ref = reference_distribution(td / "ref")
        normal = image_stats(good_dir / "0.png")
        s_normal = heuristic_score(normal, ref)
        assert min(s_normal["naturalness"], s_normal["fusion"],
                   s_normal["usability"]) >= 4, s_normal
        # 过曝图
        over = td / "over.png"
        Image.fromarray(np.clip(base_arr.astype(int) + 120, 0, 255).astype(np.uint8)
                        ).save(over)
        s_over = heuristic_score(image_stats(over), ref)
        assert s_over["naturalness"] <= 2, s_over
    print("   ✓ 正常图高分、过曝图低分")


def test_end_to_end():
    """端到端 smoke test：微型 demo 数据集 → 标注 → 改写 → 去重 → 校验 → 版本"""
    print("[7] 端到端 smoke test...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stats = build_demo_dataset(td / "ds", seed=42, categories=["bottle"],
                                   n_train_good=1, n_test_good=3,
                                   n_per_defect=2, with_duplicate=True)
        root = Path(stats["root"])
        anns = scan_mvtec_dataset(root)
        # test: good 3 + duplicate 1 + scratch 2 + stain 2 = 8
        assert len(anns) == 8, f"标注数应为 8，实际 {len(anns)}"
        assert sum(1 for a in anns if a.has_defect) == 4
        assert all(a.mask for a in anns if a.has_defect)

        recs, mode = rewrite_annotations(anns, root, {"enable_vlm": False}, 42)
        kept, dropped = dedup_by_phash(recs, root)
        assert len(dropped) == 1, f"应剔除 1 张复制图，实际 {len(dropped)}"

        from datapipe.quality import validate_records
        passed, failed = validate_records(kept, root)
        assert not failed, f"校验不应有失败: {[f['errors'] for f in failed]}"

        vdir, manifest = write_version(passed, td / "versions", {
            "pipeline": "A", "seed": 42, "config_hash": "test"})
        lines = (vdir / "dataset.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(passed) == 7
        for line in lines:
            rec = json.loads(line)
            assert set(("image", "instruction", "response")) <= set(rec)
        assert manifest["version"] == "v1" and manifest["num_records"] == 7
    print("   ✓ 端到端: 8 标注 → 去重 1 → 校验 7/7 → 版本 v1 写出")


def main() -> int:
    print("=" * 60)
    print("datapipe 数据管线测试")
    print("=" * 60)
    tests = [
        test_mask_annotation, test_defect_type_mapping, test_position_grid,
        test_dhash_dedup, test_validation, test_template_reproducibility,
        test_partial_fallback_accounting, test_heuristic_scoring,
        test_end_to_end,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"   ✗ {t.__name__} 失败: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"通过 {len(tests) - failed}/{len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
