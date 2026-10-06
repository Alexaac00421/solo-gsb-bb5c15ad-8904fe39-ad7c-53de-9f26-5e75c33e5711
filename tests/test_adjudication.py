"""复核员批量裁决（一次提交锁定/解除与隔离/恢复意图）HTTP 集成测试。

覆盖：整批按最终状态判定（意图顺序无关、同向重复只算一次、相反意图
整批拒绝）、同批解除旧锁并隔离/恢复并锁定/同批迁移锁定占用、各类整批
拒绝（引用不存在记录、解除不存在的锁定、最终锁定对非互证候选、同一
记录被多对锁定、最终既锁定又隔离、演练已封存、基准版本过期）且状态
不变、仅实际变化推进一次版本并同事务留存快照、纯重传不推进、响应契约
与既有单项接口兼容、并发批次串行化。
"""

from __future__ import annotations

import os
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("QSO_DB_PATH", tmp.name)
    from app import database  # noqa: F401
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _log(no, station, counterpart, t, band="40m", mode="SSB"):
    return {
        "record_no": no,
        "station": station,
        "counterpart": counterpart,
        "time": t,
        "band": band,
        "mode": mode,
    }


def _make_drill(client, drill="Q", allowed=10):
    r = client.post(
        "/drills", json={"drill_id": drill, "allowed_time_diff_seconds": allowed}
    )
    assert r.status_code == 201, r.text


def _seed_five(client, drill="Q", allowed=10):
    """R1-R2 配对（5s）；R3-R4 配对（8s）；R5 单边落单。批量写入后 v1。"""
    _make_drill(client, drill, allowed)
    r = client.post(f"/drills/{drill}/logs", json={"logs": [
        _log("R1", "BH1AAA", "BH2BBB", "2026-10-04T00:00:00+00:00"),
        _log("R2", "BH2BBB", "BH1AAA", "2026-10-04T00:00:05+00:00"),
        _log("R3", "BH3CCC", "BH4DDD", "2026-10-04T00:00:00+00:00"),
        _log("R4", "BH4DDD", "BH3CCC", "2026-10-04T00:00:08+00:00"),
        _log("R5", "BH5EEE", "BH6FFF", "2026-10-04T00:00:00+00:00"),
    ]})
    assert r.status_code == 200, r.text
    return r.json()["version"]


def _seed_six(client, drill="Q", allowed=10):
    """A1 可与 B1（5s）或 B2（7s）配对；C1-C2 配对（8s）；D1 单边。v1。"""
    _make_drill(client, drill, allowed)
    r = client.post(f"/drills/{drill}/logs", json={"logs": [
        _log("A1", "BH1AAA", "BH2BBB", "2026-10-04T00:00:00+00:00"),
        _log("B1", "BH2BBB", "BH1AAA", "2026-10-04T00:00:05+00:00"),
        _log("B2", "BH2BBB", "BH1AAA", "2026-10-04T00:00:07+00:00"),
        _log("C1", "BH3CCC", "BH4DDD", "2026-10-04T00:00:00+00:00"),
        _log("C2", "BH4DDD", "BH3CCC", "2026-10-04T00:00:08+00:00"),
        _log("D1", "BH5EEE", "BH6FFF", "2026-10-04T00:00:00+00:00"),
    ]})
    assert r.status_code == 200, r.text
    return r.json()["version"]


def _pair_keys(pairs):
    return sorted((p["record_a"], p["record_b"]) for p in pairs)


def _lock_keys(locks):
    return [(l["record_a"], l["record_b"]) for l in locks]


def _quarantine_nos(body):
    return [q["record_no"] for q in body["quarantine"]]


def _adjudicate(client, drill="Q", **payload):
    return client.post(f"/drills/{drill}/adjudications", json=payload)


# --------------------------------------------------------------------------
# 基本语义：混合批次、最终状态判定、整批只推进一次版本
# --------------------------------------------------------------------------
def test_mixed_batch_final_state_and_single_version_advance(client):
    _seed_five(client)  # v1

    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[
            {"action": "lock", "record_a": "R1", "record_b": "R2"},
            {"action": "lock", "record_a": "R4", "record_b": "R3"},  # 顺序任意
        ],
        quarantine_intents=[
            {"action": "quarantine", "record_no": "R5"},
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["drill_id"] == "Q"
    assert body["base_version"] == 1
    # 整批（两个锁定 + 一个隔离）只推进一次版本
    assert body["version"] == 2
    assert body["version_advanced"] is True
    # 实际生效意图按规范顺序回显（锁定对内部升序、列表升序）
    assert body["applied"] == {
        "locked": [
            {"record_a": "R1", "record_b": "R2"},
            {"record_a": "R3", "record_b": "R4"},
        ],
        "unlocked": [],
        "quarantined": ["R5"],
        "restored": [],
    }
    # 响应含当前锁定清单、隔离清单与完整互证结果
    assert _lock_keys(body["locks"]) == [("R1", "R2"), ("R3", "R4")]
    assert all(l["locked_at"] for l in body["locks"])
    assert _quarantine_nos(body) == ["R5"]
    result = body["result"]
    assert result["version"] == 2
    assert _pair_keys(result["pairs"]) == [("R1", "R2"), ("R3", "R4")]
    assert result["unpaired"] == []  # R5 被隔离，不进入未配对清单
    assert result["total_time_diff_seconds"] == 13

    # 快照同事务留存（含锁定清单与隔离清单）
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2]
    snap = client.get("/drills/Q/snapshots/2").json()
    assert _lock_keys(snap["locks"]) == [("R1", "R2"), ("R3", "R4")]
    assert _quarantine_nos(snap) == ["R5"]
    assert _pair_keys(snap["pairs"]) == [("R1", "R2"), ("R3", "R4")]

    # 既有查询接口同步体现
    got = client.get("/drills/Q/locks").json()
    assert _lock_keys(got["locks"]) == [("R1", "R2"), ("R3", "R4")]
    assert got["result"]["version"] == 2
    lst = client.get("/drills/Q/quarantine").json()
    assert _quarantine_nos(lst) == ["R5"]
    assert client.get("/drills/Q/result").json()["unpaired"] == []


def test_intent_order_does_not_change_outcome(client):
    lock_intents = [
        {"action": "lock", "record_a": "R1", "record_b": "R2"},
        {"action": "lock", "record_a": "R3", "record_b": "R4"},
    ]
    quar_intents = [{"action": "quarantine", "record_no": "R5"}]

    _seed_five(client, drill="Q1")
    b1 = _adjudicate(
        client, drill="Q1", base_version=1,
        lock_intents=lock_intents, quarantine_intents=quar_intents,
    ).json()
    _seed_five(client, drill="Q2")
    b2 = _adjudicate(
        client, drill="Q2", base_version=1,
        lock_intents=list(reversed(lock_intents)),
        quarantine_intents=quar_intents,
    ).json()

    # 意图顺序不影响结果：生效意图、清单与结果完全一致（时间戳除外）
    assert b1["applied"] == b2["applied"]
    assert b1["version"] == b2["version"] == 2
    assert _lock_keys(b1["locks"]) == _lock_keys(b2["locks"])
    assert _quarantine_nos(b1) == _quarantine_nos(b2)
    res1 = {k: v for k, v in b1["result"].items()
            if k not in ("drill_id", "recomputed_at")}
    res2 = {k: v for k, v in b2["result"].items()
            if k not in ("drill_id", "recomputed_at")}
    assert res1 == res2


def test_duplicate_same_direction_intents_count_once(client):
    _seed_five(client)  # v1

    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[
            {"action": "lock", "record_a": "R1", "record_b": "R2"},
            {"action": "lock", "record_a": "R2", "record_b": "R1"},  # 同向重复
            {"action": "lock", "record_a": " R1 ", "record_b": "R2"},  # 空白差异
        ],
        quarantine_intents=[
            {"action": "quarantine", "record_no": "R5"},
            {"action": "quarantine", "record_no": " R5 "},  # 同向重复
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 同向重复只算一次：生效清单各出现一次，整批仍只推进一次版本
    assert body["applied"]["locked"] == [{"record_a": "R1", "record_b": "R2"}]
    assert body["applied"]["quarantined"] == ["R5"]
    assert body["version"] == 2
    assert _lock_keys(body["locks"]) == [("R1", "R2")]
    assert _quarantine_nos(body) == ["R5"]


def test_empty_batch_is_noop_retransmit(client):
    _seed_five(client)  # v1

    r = _adjudicate(client, base_version=1)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1
    assert body["version_advanced"] is False
    assert body["applied"] == {
        "locked": [], "unlocked": [], "quarantined": [], "restored": []
    }
    assert body["locks"] == [] and body["quarantine"] == []
    assert body["result"]["version"] == 1
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1]


# --------------------------------------------------------------------------
# 相互牵连的裁决：同批解除并隔离 / 恢复并锁定 / 迁移锁定占用
# --------------------------------------------------------------------------
def test_unlock_old_lock_and_quarantine_same_batch(client):
    _seed_five(client)
    client.put("/drills/Q/locks", json={"record_a": "R1", "record_b": "R2"})  # v2

    # 同批解除旧锁并隔离其成员：按最终状态判定，合法（单项接口须分两步）
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "unlock", "record_a": "R2", "record_b": "R1"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "R1"}],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 3
    assert body["applied"]["unlocked"] == [{"record_a": "R1", "record_b": "R2"}]
    assert body["applied"]["quarantined"] == ["R1"]
    assert body["locks"] == []
    assert _quarantine_nos(body) == ["R1"]
    # R1 被隔离后其对方 R2 落单；R3-R4 继续配对
    assert _pair_keys(body["result"]["pairs"]) == [("R3", "R4")]
    assert body["result"]["unpaired"] == ["R2", "R5"]


def test_restore_and_lock_same_batch(client):
    _seed_five(client)
    client.put("/drills/Q/quarantine", json={"record_nos": ["R2"]})  # v2

    # 同批恢复并锁定：最终状态中 R2 未隔离，锁定合法
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "restore", "record_no": "R2"}],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 3
    assert body["applied"]["restored"] == ["R2"]
    assert body["applied"]["locked"] == [{"record_a": "R1", "record_b": "R2"}]
    assert body["quarantine"] == []
    assert _lock_keys(body["locks"]) == [("R1", "R2")]
    # 锁定对强制入选
    assert ("R1", "R2") in _pair_keys(body["result"]["pairs"])


def test_move_lock_to_overlapping_pair_same_batch(client):
    _seed_six(client)  # v1
    client.put("/drills/Q/locks", json={"record_a": "A1", "record_b": "B1"})  # v2

    # 同批解除旧锁、用同一记录锁定新对：最终状态每条记录至多一对，合法
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[
            {"action": "unlock", "record_a": "A1", "record_b": "B1"},
            {"action": "lock", "record_a": "A1", "record_b": "B2"},
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 3
    assert body["applied"]["unlocked"] == [{"record_a": "A1", "record_b": "B1"}]
    assert body["applied"]["locked"] == [{"record_a": "A1", "record_b": "B2"}]
    assert _lock_keys(body["locks"]) == [("A1", "B2")]
    pairs = _pair_keys(body["result"]["pairs"])
    assert ("A1", "B2") in pairs  # 新锁定对强制入选（即使总时差更大）
    assert ("A1", "B1") not in pairs


def test_unlock_and_relock_different_pairs_same_batch(client):
    _seed_five(client)
    client.put("/drills/Q/locks", json={"record_a": "R1", "record_b": "R2"})  # v2

    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[
            {"action": "unlock", "record_a": "R1", "record_b": "R2"},
            {"action": "lock", "record_a": "R3", "record_b": "R4"},
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 3
    assert _lock_keys(body["locks"]) == [("R3", "R4")]
    assert _pair_keys(body["result"]["pairs"]) == [
        ("R1", "R2"), ("R3", "R4")
    ]


# --------------------------------------------------------------------------
# 整批拒绝：相反意图 / 记录缺失 / 解除不存在的锁定 / 非互证候选 /
#           多对锁定同一记录 / 最终既锁定又隔离 / 已封存 / 基准版本过期
# --------------------------------------------------------------------------
def test_opposite_intents_rejected_and_state_unchanged(client):
    _seed_five(client)
    client.put("/drills/Q/locks", json={"record_a": "R1", "record_b": "R2"})  # v2
    client.put("/drills/Q/quarantine", json={"record_nos": ["R5"]})  # v3

    # 同一记录对既解除又锁定（含颠倒顺序）
    r = _adjudicate(
        client,
        base_version=3,
        lock_intents=[
            {"action": "unlock", "record_a": "R1", "record_b": "R2"},
            {"action": "lock", "record_a": "R2", "record_b": "R1"},
        ],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjudication_intent_conflict"

    # 同一记录既恢复又隔离
    r = _adjudicate(
        client,
        base_version=3,
        quarantine_intents=[
            {"action": "restore", "record_no": "R5"},
            {"action": "quarantine", "record_no": "R5"},
        ],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjudication_intent_conflict"

    # 状态不变：版本、锁定、隔离、快照均保持
    assert client.get("/drills/Q/result").json()["version"] == 3
    assert _lock_keys(client.get("/drills/Q/locks").json()["locks"]) == [
        ("R1", "R2")
    ]
    assert _quarantine_nos(client.get("/drills/Q/quarantine").json()) == ["R5"]
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2, 3]


def test_missing_records_rejected_and_state_unchanged(client):
    _seed_five(client)  # v1

    # 锁定意图引用不存在的记录
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "NOPE"}],
    )
    assert r.status_code == 404
    assert r.json()["error"] == "lock_record_missing"

    # 隔离/恢复意图引用不存在的记录
    for action in ("quarantine", "restore"):
        r = _adjudicate(
            client,
            base_version=1,
            quarantine_intents=[{"action": action, "record_no": "NOPE"}],
        )
        assert r.status_code == 404
        assert r.json()["error"] == "quarantine_record_missing"

    # 混合批次中任一引用缺失即整批拒绝
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "NOPE"}],
    )
    assert r.status_code == 404
    assert r.json()["error"] == "quarantine_record_missing"

    assert client.get("/drills/Q/result").json()["version"] == 1
    assert client.get("/drills/Q/locks").json()["locks"] == []
    assert client.get("/drills/Q/quarantine").json()["quarantine"] == []
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1]


def test_unlock_nonexistent_lock_rejected(client):
    _seed_five(client)  # v1（无任何锁定）

    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "unlock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 404
    assert r.json()["error"] == "lock_not_found"
    assert client.get("/drills/Q/result").json()["version"] == 1
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1]


def test_final_lock_not_candidate_rejected(client):
    _seed_five(client)  # v1

    # R1 与 R3 未相互指认
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R3"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "lock_not_candidate"

    # 时差超限：写入一对互指但时差 60 秒（允许 10 秒）的记录
    client.post("/drills/Q/logs", json={"logs": [
        _log("T1", "BH7GGG", "BH8HHH", "2026-10-04T00:00:00+00:00"),
        _log("T2", "BH8HHH", "BH7GGG", "2026-10-04T00:01:00+00:00"),
    ]})  # v2
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "lock", "record_a": "T1", "record_b": "T2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "lock_not_candidate"

    assert client.get("/drills/Q/result").json()["version"] == 2
    assert client.get("/drills/Q/locks").json()["locks"] == []


def test_same_record_locked_by_multiple_pairs_rejected(client):
    _seed_six(client)  # v1；A1 与 B1/B2 均为合法候选

    # 批次内两个新锁定占用同一记录
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[
            {"action": "lock", "record_a": "A1", "record_b": "B1"},
            {"action": "lock", "record_a": "A1", "record_b": "B2"},
        ],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "record_already_locked"

    # 新锁定与既有锁定（未同批解除）占用同一记录
    client.put("/drills/Q/locks", json={"record_a": "A1", "record_b": "B1"})  # v2
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "lock", "record_a": "A1", "record_b": "B2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "record_already_locked"

    assert client.get("/drills/Q/result").json()["version"] == 2
    assert _lock_keys(client.get("/drills/Q/locks").json()["locks"]) == [
        ("A1", "B1")
    ]


def test_lock_quarantine_conflict_rejected(client):
    _seed_five(client)  # v1

    # 同批锁定并隔离同一记录：最终状态既锁定又隔离
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "R1"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "lock_quarantine_conflict"

    # 既有锁定成员不同批解除直接隔离：最终仍既锁定又隔离
    client.put("/drills/Q/locks", json={"record_a": "R1", "record_b": "R2"})  # v2
    r = _adjudicate(
        client,
        base_version=2,
        quarantine_intents=[{"action": "quarantine", "record_no": "R2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "lock_quarantine_conflict"

    # 既有隔离记录不同批恢复直接锁定：同样冲突
    client.post("/drills/Q/locks/unlock",
                json={"record_a": "R1", "record_b": "R2"})  # v3
    client.put("/drills/Q/quarantine", json={"record_nos": ["R1"]})  # v4
    r = _adjudicate(
        client,
        base_version=4,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "lock_quarantine_conflict"

    # 全部拒绝后状态不变
    assert client.get("/drills/Q/result").json()["version"] == 4
    assert client.get("/drills/Q/locks").json()["locks"] == []
    assert _quarantine_nos(client.get("/drills/Q/quarantine").json()) == ["R1"]
    assert client.get("/drills/Q/snapshots").json()["versions"] == [
        0, 1, 2, 3, 4
    ]


def test_stale_base_version_rejected(client):
    _seed_five(client)  # v1

    r = _adjudicate(
        client,
        base_version=0,  # 当前已是 v1
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjudication_base_version_changed"

    # 推进版本后旧基准版本同样过期（即使批次本身为纯重传）
    client.put("/drills/Q/locks", json={"record_a": "R1", "record_b": "R2"})  # v2
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjudication_base_version_changed"

    assert client.get("/drills/Q/result").json()["version"] == 2
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2]


def test_sealed_drill_rejected(client):
    _seed_five(client)
    client.post("/drills/Q/seal")

    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 409
    assert r.json()["error"] == "drill_sealed"
    assert client.get("/drills/Q/result").json()["version"] == 1


def test_unknown_drill_rejected(client):
    r = _adjudicate(
        client, drill="NOPE",
        base_version=0,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    )
    assert r.status_code == 404
    assert r.json()["error"] == "drill_not_found"


def test_request_validation_errors(client):
    _seed_five(client)

    # 同一记录号组成的对（含空白差异）-> 422 lock_same_record
    r = _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": " R1 "}],
    )
    assert r.status_code == 422
    assert r.json()["error"] == "lock_same_record"

    # 非法动作 / 空记录号 / 缺 base_version -> FastAPI 标准 422
    assert _adjudicate(
        client, base_version=1,
        lock_intents=[{"action": "freeze", "record_a": "R1", "record_b": "R2"}],
    ).status_code == 422
    assert _adjudicate(
        client, base_version=1,
        quarantine_intents=[{"action": "quarantine", "record_no": "  "}],
    ).status_code == 422
    assert _adjudicate(
        client,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
    ).status_code == 422

    # 校验失败不产生任何状态变化
    assert client.get("/drills/Q/result").json()["version"] == 1
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1]


# --------------------------------------------------------------------------
# 纯重传不推进版本；快照与比较反映批次变化
# --------------------------------------------------------------------------
def test_pure_retransmit_does_not_advance(client):
    _seed_five(client)  # v1
    _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "R5"}],
    )  # v2

    # 全部意图均不改变状态：锁定已存在、隔离已存在、恢复未隔离记录
    r = _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "lock", "record_a": "R2", "record_b": "R1"}],
        quarantine_intents=[
            {"action": "quarantine", "record_no": "R5"},
            {"action": "restore", "record_no": "R3"},
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 2
    assert body["version_advanced"] is False
    assert body["applied"] == {
        "locked": [], "unlocked": [], "quarantined": [], "restored": []
    }
    # 清单与结果照常返回
    assert _lock_keys(body["locks"]) == [("R1", "R2")]
    assert _quarantine_nos(body) == ["R5"]
    # 不推进版本、不留存新快照
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2]


def test_snapshot_and_diff_reflect_batch(client):
    _seed_five(client)  # v1
    _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "R5"}],
    )  # v2
    _adjudicate(
        client,
        base_version=2,
        lock_intents=[{"action": "unlock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "restore", "record_no": "R5"}],
    )  # v3

    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2, 3]
    v2 = client.get("/drills/Q/snapshots/2").json()
    assert _lock_keys(v2["locks"]) == [("R1", "R2")]
    assert _quarantine_nos(v2) == ["R5"]
    v3 = client.get("/drills/Q/snapshots/3").json()
    assert v3["locks"] == [] and v3["quarantine"] == []

    # 版本比较：锁定与隔离的新增/移除按身份单列
    d = client.get("/drills/Q/snapshots/1/diff/2").json()
    assert _lock_keys(d["locks"]["added"]) == [("R1", "R2")]
    assert d["locks"]["removed"] == []
    assert _quarantine_nos({"quarantine": d["quarantine"]["added"]}) == ["R5"]
    d = client.get("/drills/Q/snapshots/2/diff/3").json()
    assert _lock_keys(d["locks"]["removed"]) == [("R1", "R2")]
    assert _quarantine_nos({"quarantine": d["quarantine"]["removed"]}) == ["R5"]

    # 历史快照保持原貌：后续批次不改写已留存内容
    assert client.get("/drills/Q/snapshots/2").json() == v2


# --------------------------------------------------------------------------
# 与既有单项接口兼容；并发批次串行化
# --------------------------------------------------------------------------
def test_single_item_endpoints_still_work_after_batch(client):
    _seed_five(client)  # v1
    _adjudicate(
        client,
        base_version=1,
        lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        quarantine_intents=[{"action": "quarantine", "record_no": "R5"}],
    )  # v2

    # 单项解除锁定：响应契约不变，版本照常 +1
    r = client.post("/drills/Q/locks/unlock",
                    json={"record_a": "R2", "record_b": "R1"})
    assert r.status_code == 200
    assert r.json()["locks"] == []
    assert r.json()["result"]["version"] == 3

    # 单项恢复隔离：响应契约不变，版本照常 +1
    r = client.post("/drills/Q/quarantine/restore",
                    json={"record_nos": ["R5"]})
    assert r.status_code == 200
    assert r.json()["affected_record_nos"] == ["R5"]
    assert r.json()["quarantine"] == []
    assert r.json()["result"]["version"] == 4

    # 单项锁定：响应契约不变
    r = client.put("/drills/Q/locks", json={"record_a": "R3", "record_b": "R4"})
    assert r.status_code == 200
    assert r.json()["lock"]["record_a"] == "R3"
    assert r.json()["result"]["version"] == 5


def test_concurrent_adjudications_serialized(client):
    _seed_five(client)  # v1
    outcomes = []

    def adj():
        r = _adjudicate(
            client,
            base_version=1,
            lock_intents=[{"action": "lock", "record_a": "R1", "record_b": "R2"}],
        )
        outcomes.append((r.status_code, r.json().get("error")))

    threads = [threading.Thread(target=adj) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 写事务串行化：仅第一个批次按基准版本 1 受理，其余因版本过期拒绝
    assert outcomes.count((200, None)) == 1
    assert outcomes.count((409, "adjudication_base_version_changed")) == 4
    assert client.get("/drills/Q/result").json()["version"] == 2
    assert _lock_keys(client.get("/drills/Q/locks").json()["locks"]) == [
        ("R1", "R2")
    ]
    assert client.get("/drills/Q/snapshots").json()["versions"] == [0, 1, 2]
