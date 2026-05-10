from __future__ import annotations

from pathlib import Path

from research_agent.domain import DomainPack


def get_domain_pack() -> DomainPack:
    """CosmoTwin extension sketch for scientific research.

    This pack intentionally does not register trading tools or trading metrics.
    Add CosmoTwin tools such as paper search, simulation runners, telescope data
    loaders, model-fitting evaluators, and uncertainty metrics here while
    reusing the same graph kernel, memory, and experiment logger.
    """

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    return DomainPack(
        name="cosmotwin",
        description="Scientific literature, simulation, and model-validation research.",
        prompt_dir=prompt_dir,
        metrics={
            "primary": ["reproducibility", "uncertainty_quality", "citation_quality"],
            "validation": ["simulation_fit", "physical_plausibility", "ablation_coverage"],
        },
        risk_limits={
            "min_citation_quality": 0.75,
            "min_reproducibility_score": 0.80,
        },
        report_warning=(
            "CosmoTwin scientific-research output. Requires domain expert review, "
            "reproducibility checks, and independent validation before publication or operational use."
        ),
    )
