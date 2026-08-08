from __future__ import annotations

import json

import pytest
import torch

import nablaguard as ng


def test_custom_contract_emits_normal_issue() -> None:
    assertion = ng.contract(
        "positive-loss", lambda context: context.loss is not None and context.loss > 0
    )
    with ng.Session() as session:
        issue = assertion(ng.ContractContext(loss=-1.0))

    assert issue is session.issues[0]
    assert issue.code == "NG5001"
    assert issue.evidence["contract"] == "positive-loss"


def test_loss_contract_can_raise() -> None:
    assertion = ng.contracts.loss.finite(raise_on_failure=True)
    with pytest.raises(ng.ContractViolation, match="NG1001"):
        assertion(ng.ContractContext(loss=torch.tensor(float("nan"))))


def test_tensor_contract_integrates_with_guard() -> None:
    with ng.guard(
        dispatch=False,
        contracts=[ng.contracts.tensor.finite(module="encoder.*")],
    ) as monitor:
        monitor.observe(torch.tensor([float("inf")]), module_path="decoder.0")
        monitor.observe(torch.tensor([float("inf")]), module_path="encoder.0")

    contract_issues = [issue for issue in monitor.issues if issue.evidence.get("contract")]
    assert len(contract_issues) == 1
    assert contract_issues[0].module_path == "encoder.0"


def test_gradient_and_parameter_contracts() -> None:
    gradient = ng.contracts.gradient.norm(max=1.0)
    parameter = ng.contracts.parameter.change(min_relative=0.1, max_relative=0.3)
    context = ng.ContractContext(
        gradients={"weight": torch.tensor([3.0, 4.0])},
        parameters={"weight": torch.tensor([1.2])},
        previous_parameters={"weight": torch.tensor([1.0])},
    )

    assert gradient(context) is not None
    assert parameter(context) is None


def test_capture_evaluates_contracts_and_persists_issues(tmp_path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with ng.capture(
        model,
        root=tmp_path,
        run_id="contracts",
        checkpoint_every=10,
        contracts=[
            ng.contracts.loss.finite(),
            ng.contracts.training.loss_not_exploding(max_ratio=2.0),
        ],
    ) as recorder:
        first = recorder.record_step(step=1, loss=1.0)
        second = recorder.record_step(step=2, loss=3.0)

    assert first is not None and second is not None
    first_data = json.loads(first.read_text(encoding="utf-8"))
    second_data = json.loads(second.read_text(encoding="utf-8"))
    assert first_data["contract_issues"] == []
    assert second_data["contract_issues"][0]["category"] == "TRAINING_DIVERGENCE"
    assert len(recorder.contract_issues) == 1


def test_parameter_contract_observes_step_change(tmp_path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with ng.capture(
        model,
        root=tmp_path,
        run_id="parameter-contract",
        checkpoint_every=10,
        contracts=[ng.contracts.parameter.change(min_relative=0.01)],
    ) as recorder:
        model.weight.data.mul_(1.5)
        recorder.record_step(step=1, loss=1.0)

    assert recorder.contract_issues == []
