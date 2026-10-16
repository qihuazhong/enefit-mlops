"""The SageMaker Pipeline runs the steps in a chain, so the registry numbers stay v1..v6 = #1..#6 and the live run
starts from v6 (#6, @champion)."""
from enefit.pipeline import definition


def test_steps_run_in_a_chain():
    steps = definition("image:abc1234", "role", "bucket", "mlflow-arn")["Steps"]
    assert [s["Name"] for s in steps] == ["features", "v1", "v2", "v3", "v4", "v5", "v6", "live-run"]
    assert "DependsOn" not in steps[0]
    assert all(s["DependsOn"] == [prev["Name"]] for prev, s in zip(steps, steps[1:]))
    args = {s["Name"]: s["Arguments"]["AlgorithmSpecification"]["ContainerArguments"] for s in steps[1:]}
    assert [name for name, a in args.items() if "--champion" in a] == ["v6"]
    assert args["live-run"][:2] == ["-m", "enefit.live_run"]
