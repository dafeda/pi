"""Check loopback domain operations and Pi-to-Python calls with prewritten responses."""

import http.client
import json
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from cases.pressure import PressureCase
from controller.campaign import Campaign
from controller.experiment import evaluate_selection, prepare_case
from controller.server import ScientificServer
from numerical.geology import BandGeometry, Family


def request(server: ScientificServer, path: str, body: object = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    message = Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(message, timeout=60) as response:
        return json.load(response)


@pytest.fixture
def service(tmp_path: Path):
    case = PressureCase(grid_size=8, segment_steps=4, sample_every=1)
    prepare_case(
        tmp_path / "snapshot",
        case,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    with Campaign(
        tmp_path / "snapshot" / "evidence", tmp_path / "campaign", members=16
    ) as campaign:
        server = ScientificServer(campaign, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


def test_loopback_http_and_domain_policy(service: ScientificServer) -> None:
    assert service.server_address[0] == "127.0.0.1"
    with pytest.raises(HTTPError) as no_files:
        request(service, "/protected/truth.json")
    assert no_files.value.code == 404
    no_files.value.close()
    with pytest.raises(HTTPError) as invalid:
        request(
            service,
            "/v1/evaluations",
            {"hypothesis": {"kind": "homogeneous"}, "reason": "test", "python_code": "read truth"},
        )
    assert invalid.value.code == 400
    invalid.value.close()
    assert len(request(service, "/v1/campaign")["evaluations"]) == 1
    evidence = request(service, "/v1/evidence")
    assert "band_width_fraction" not in evidence["physics"]
    assert "implementation" not in evidence


def test_losing_an_http_wait_does_not_cancel_accepted_work(service: ScientificServer) -> None:
    accepted = request(
        service,
        "/v1/evaluations",
        {
            "hypothesis": {
                "kind": "band",
                "orientation": "vertical",
                "center_fraction": 0.5,
                "width_fraction": 0.25,
            },
            "reason": "Accepted work must outlive a client connection",
        },
    )
    connection = http.client.HTTPConnection("127.0.0.1", service.server_port)
    connection.request("GET", f"/v1/runs/{accepted['run_id']}/wait")
    connection.close()
    completed = request(service, f"/v1/runs/{accepted['run_id']}/wait")
    assert completed["status"] == "succeeded"
    assert completed["result"]["adequacy_gate_passed"] is True


@pytest.mark.parametrize(
    ("family", "orientation"),
    [
        (Family.VERTICAL_BAND, "vertical"),
        (Family.HORIZONTAL_BAND, "horizontal"),
        (Family.HOMOGENEOUS, "vertical"),
    ],
)
def test_pi_scripted_round_trip_and_homogeneous_negative_control(
    tmp_path: Path, family: Family, orientation: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if (
        not node
        or not (project / "agent" / "node_modules" / "@earendil-works" / "pi-agent-core").exists()
    ):
        pytest.skip("Pi bridge test requires Node >=22.19 and the agent dependency setup")
    version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout
    major, minor, *_patch = map(int, version.strip().lstrip("v").split("."))
    if major < 22 or (major == 22 and minor < 19):
        pytest.skip("Pi bridge test requires Node >=22.19")
    case = PressureCase(grid_size=8, segment_steps=4, sample_every=1)
    snapshot = tmp_path / "snapshot"
    prepare_case(
        snapshot,
        case,
        family,
        truth_geometry=BandGeometry(0.5, 0.25),
    )
    run = tmp_path / "campaign"
    with Campaign(snapshot / "evidence", run, members=16) as campaign:
        server = ScientificServer(campaign, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            process = subprocess.run(
                [
                    node,
                    "--experimental-strip-types",
                    "src/cli.ts",
                    "--faux",
                    "--faux-orientation",
                    orientation,
                    "--url",
                    f"http://127.0.0.1:{server.server_port}/",
                    "--audit-dir",
                    str(tmp_path / "audit"),
                ],
                cwd=project / "agent",
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert process.returncode == 0, process.stderr
            output = json.loads(process.stdout)
            assert output["mode"] == "faux_scripted_smoke_test"
            state = campaign.snapshot()
            expected = 1 if family == Family.HOMOGENEOUS else 2
            assert state["closed"] and state["outcome"] == "adequate"
            assert len(state["evaluations"]) == expected
            assert state["model_calls"] == expected
            assert state["charged_simulator_calls"] == expected * 80
            transcript = (tmp_path / "audit" / "messages.jsonl").read_text()
            assert "truth_family" not in transcript
            assert "band_width_fraction" not in transcript
            assert str(snapshot / "protected") not in transcript
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
    report = evaluate_selection(
        snapshot / "evidence", snapshot / "protected", run, tmp_path / "grade"
    )
    assert report["structural_truth_recovered"] is True
