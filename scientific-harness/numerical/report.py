"""Render protected post-selection diagnostics with a headless Matplotlib backend."""

from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from cases.pressure import PressureCase
from numerical.geology import MILLIDARCY_M2


def diagnostic_png(
    case: PressureCase,
    truth_field_m2: NDArray[np.float64],
    posterior_field_m2: NDArray[np.float64],
    clean_responses_pa: NDArray[np.float64],
    predictions_pa: NDArray[np.float64],
    observations_pa: NDArray[np.float64],
    selected_family: str,
) -> bytes:
    """Compare truth/posterior fields and 90% predictive intervals at all four gauges."""
    figure = Figure(figsize=(11, 10), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    fields = [np.log10(field / MILLIDARCY_M2) for field in (truth_field_m2, posterior_field_m2)]
    vmin = min(float(field.min()) for field in fields)
    vmax = max(float(field.max()) for field in fields)
    pattern = selected_family.replace("_", " ")
    for axis, field, title in zip(
        axes[0],
        fields,
        (
            "Hidden truth",
            f"Conditioned {pattern}",
        ),
        strict=True,
    ):
        image = axis.imshow(
            field,
            origin="lower",
            extent=(0, case.length_m, 0, case.length_m),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        gauges = case.gauge_positions * case.length_m
        axis.scatter(gauges[:, 0], gauges[:, 1], marker="x", color="white", label="gauges")
        axis.set(title=title, xlabel="x (m)", ylabel="y (m)")
        figure.colorbar(
            image,
            ax=axis,
            label="log10(permeability / mD)",
        )

    times_h = case.observation_steps * case.time_step_s / 3600
    shape = (times_h.size, len(case.gauge_ids))
    truth = (clean_responses_pa.reshape(shape) - case.initial_pressure_pa) / 1e5
    observed = (observations_pa.reshape(shape) - case.initial_pressure_pa) / 1e5
    predictions = (
        predictions_pa.reshape(*shape, predictions_pa.shape[1]) - case.initial_pressure_pa
    ) / 1e5
    lower, upper = np.quantile(predictions, [0.05, 0.95], axis=2)
    for gauge, axis in enumerate(axes[1:].flat):
        axis.fill_between(
            times_h,
            lower[:, gauge],
            upper[:, gauge],
            alpha=0.25,
            color="tab:blue",
            label="90% predictive interval",
        )
        axis.plot(times_h, predictions[:, gauge].mean(axis=1), label="predictive mean")
        axis.plot(
            times_h,
            truth[:, gauge],
            color="black",
            linestyle="--",
            label="noise-free truth",
        )
        axis.scatter(times_h, observed[:, gauge], s=10, color="tab:orange", label="observations")
        for segments, label in ((4, "conditioning ends"), (6, "validation ends")):
            axis.axvline(
                segments * case.segment_steps * case.time_step_s / 3600, color="gray", label=label
            )
        axis.set(
            title=case.gauge_ids[gauge].replace("_", " "),
            xlabel="time (hours)",
            ylabel="pressure change (bar)",
        )
    axes[1, 0].legend(fontsize="small")
    figure.suptitle(
        "Permeability and pressure forecasts\nConditioning | validation | protected final"
    )
    output = BytesIO()
    canvas.print_png(output)
    return output.getvalue()
