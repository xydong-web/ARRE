"""Read the accepted same-oracle comparison without recomputing inference."""
import hashlib
import json

RECEIPT = "acceptance/supp_results_all_20261107.json"
SHA256 = "9ceea3bf8413e7580f2ead69ac2989bc813fe95911ce199b612732707133a13b"
DEV_METHODS = (
    ("B", "CGFlow"), ("F", "CGFlow + ARRE"),
    ("RX", "RxnFlow"), ("RXCE", "RxnFlow + ARRE"),
    ("SFN", "SynFlowNet"), ("GA", "GraphGA"),
)


def accepted(data_dir):
    payload = (data_dir / RECEIPT).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SHA256
    report = json.loads(payload)
    for cohort, n, arms in (("dev15", 15, dict(DEV_METHODS)),
                            ("five5", 5, ("B", "F", "RX", "S3"))):
        panel = report[cohort]
        assert panel["n_pockets"] == n and not panel["missing"]
        assert panel["n_cells"] == dict.fromkeys(arms, n)
        assert set(panel["means"]) == set(arms)
    return report
