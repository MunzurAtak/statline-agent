import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from statline_agent.agent_eval import (
    AgentCase,
    CaseResult,
    Expected,
    Judgement,
    ReferenceQuery,
    UsageCounter,
    compute_reference,
    extract_numbers,
    judge_answer,
    judge_context,
    mentions,
    reference_value,
    summarize,
    unit_scale,
)
from statline_agent.cbs import CBSClient
from tests.fakes import FakeStructuredChatModel, prompt_text


def test_extract_numbers_in_dutch_notation() -> None:
    text = "Rotterdam: 673.804 inwoners, Utrecht 1 415 104, groei 56,1% en circa 1,4 miljoen."
    assert extract_numbers(text) == [
        (673804.0, False),
        (1415104.0, False),
        (56.1, False),
        (1400000.0, True),
    ]


def test_mentions_with_tolerance_and_units() -> None:
    assert mentions("Er wonen 673.804 mensen.", Expected(673804))
    assert mentions("Er wonen ruim 674.000 mensen.", Expected(673804))  # rounded
    assert not mentions("Er wonen 672.960 mensen.", Expected(673804))  # last year's figure
    assert mentions("Ongeveer 1,4 miljoen inwoners.", Expected(1415104))  # approximate wording
    assert mentions("Gestegen met 56,1%.", Expected(56.107, is_percentage=True))
    # WOZ is published in x 1 000 euro: the answer has to apply the unit
    assert mentions("De WOZ-waarde is €374.000.", Expected(374, scale=1000))
    assert mentions("De WOZ-waarde is 374 duizend euro.", Expected(374, scale=1000))
    assert not mentions("De WOZ-waarde is €374.", Expected(374, scale=1000))
    assert not mentions("De WOZ-waarde is 374.00 euro.", Expected(374, scale=1000))
    # ...but a percentage change is not scaled
    assert mentions("Gestegen met 56,1%.", Expected(56.107, is_percentage=True, scale=1000))


def test_expected_target_applies_the_unit() -> None:
    assert Expected(374, scale=1000).target == 374_000
    assert Expected(56.1, is_percentage=True, scale=1000).target == 56.1


def test_unit_scale() -> None:
    assert unit_scale("x 1 000 euro") == 1000
    assert unit_scale("mln euro") == 1_000_000
    assert unit_scale("per 1 000 inwoners") == 1  # a rate, not a multiplier
    assert unit_scale("aantal") == 1


def test_reference_value_kinds() -> None:
    values = {"2015JJ00": 100.0, "2020JJ00": 110.0, "2025JJ00": 121.0}
    ref = ReferenceQuery(table_id="T", measure="M")
    assert reference_value(ref, values) == 121.0
    assert reference_value(ref.model_copy(update={"kind": "year", "year": 2020}), values) == 110.0
    growth = ref.model_copy(update={"kind": "change_pct", "start_year": 2015, "end_year": 2025})
    assert reference_value(growth, values) == pytest.approx(21.0)
    since = ref.model_copy(update={"kind": "change_pct", "start_year": 2020})
    assert reference_value(since, values) == pytest.approx(10.0)


def test_compute_reference_on_recorded_cbs_data(cbs_client: CBSClient) -> None:
    ref = ReferenceQuery(
        table_id="70072ned", measure="Totale bevolking", filters={"RegioS": ["Zuid-Holland"]}
    )
    expected = compute_reference(cbs_client, ref)
    assert expected == Expected(value=598239.0, is_percentage=False, scale=1.0)


def test_compute_reference_reports_unknown_measure(cbs_client: CBSClient) -> None:
    with pytest.raises(ValueError, match="not found"):
        compute_reference(cbs_client, ReferenceQuery(table_id="70072ned", measure="Bestaat niet"))


def test_judge_sees_result_context_and_answer() -> None:
    llm = FakeStructuredChatModel(responses=[Judgement(grounded=True, issues=[])])
    verdict = judge_answer(
        llm, "Hoeveel?", ["n"], [{"n": 5.0}], "Het zijn er 5.", context="Tabel: T1 Test"
    )
    assert verdict.grounded
    sent = prompt_text(llm.prompts[0])
    assert "n\n5.00" in sent and "Het zijn er 5." in sent and "Tabel: T1 Test" in sent


def test_judge_verdict_follows_the_issues() -> None:
    # A small judge model may contradict itself; the list of issues is what counts.
    llm = FakeStructuredChatModel(
        responses=[
            Judgement(grounded=False, issues=[]),
            Judgement(grounded=True, issues=["Het getal 6 staat niet in het resultaat."]),
        ]
    )
    assert judge_answer(llm, "?", ["n"], [{"n": 5.0}], "5").grounded
    assert not judge_answer(llm, "?", ["n"], [{"n": 5.0}], "6").grounded


def test_judge_context_lists_table_measures_units_and_filters() -> None:
    state = {
        "table_id": "85036NED",
        "table_title": "Woningwaarde",
        "measure_titles": ["Woningen > Gemiddelde WOZ-waarde"],
        "measure_units": ["x 1 000 euro"],
        "requested_filters": {"Regio's": ["Den Haag"]},
    }
    assert judge_context(state) == (
        "Tabel: 85036NED Woningwaarde\n"
        "Onderwerp: Woningen > Gemiddelde WOZ-waarde (eenheid: x 1 000 euro)\n"
        "Regio's: Den Haag"
    )


def test_usage_counter_counts_calls_and_tokens() -> None:
    counter = UsageCounter()
    message = AIMessage(
        "x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert (counter.calls, counter.tokens) == (1, 15)


def test_summarize() -> None:
    results = [
        CaseResult("a", "q", table_ok=True, answer_ok=True, grounded=True, attempts=1, seconds=4),
        CaseResult("b", "q", table_ok=False, answer_ok=False, grounded=None, seconds=2),
    ]
    summary = summarize(results)
    assert summary["table_accuracy"] == 0.5
    assert summary["answer_accuracy"] == 0.5
    assert summary["groundedness"] == 1.0
    assert summary["avg_sql_attempts"] == 1.0
    assert summary["median_seconds"] == 3.0


def test_repository_cases_are_valid() -> None:
    path = Path(__file__).parents[1] / "evals" / "agent_cases.json"
    cases = [AgentCase.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
    assert len({case.id for case in cases}) == len(cases) >= 10
    for case in cases:
        assert case.turns
        # Cases that expect data must say how to compute the right answer.
        assert bool(case.expected_tables) == bool(case.references), case.id
