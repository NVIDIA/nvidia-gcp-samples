"""Run with python3 -m unittest discover -s tests; no cloud credentials needed."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SAMPLE = Path(__file__).resolve().parents[1]
NAME = "rfd4-test"
PROJECT = "test-project"
REGION = "us-central1"

# Exercise the real shell scripts with stateful cloud/Kubernetes command doubles.
COMMAND = r'''
import json
import os
from pathlib import Path
import sys

path = Path(os.environ["TEST_STATE"])
state = json.loads(path.read_text())
tool, args = Path(sys.argv[0]).name, sys.argv[1:]
state["calls"].append([tool, *args])
output, error, code = "", "", 0
if tool == "gcloud":
    kinds = {
        ("container", "clusters"): "cluster",
        ("artifacts", "repositories"): "repository",
        ("compute", "networks", "subnets"): "subnet",
        ("compute", "networks"): "network",
    }
    kind = next((v for k, v in kinds.items() if tuple(args[:len(k)]) == k), None)
    if "list" in args and kind:
        if state.get("fail_lookup") == kind:
            code, error = 1, state.get("lookup_error", "PERMISSION_DENIED")
        elif state.get(kind):
            output = "rfd4-test\n"
    elif "delete" in args and kind:
        if kind == "cluster" and state.get("delete_busy"):
            state["delete_busy"] = False
            code, error = 1, "incompatible operation"
        else:
            state[kind] = False
    elif args[:3] == ["container", "clusters", "get-credentials"]:
        pass
    else:
        code, error = 1, "Unexpected gcloud command: " + str(args)
elif tool == "kubectl":
    if args[:2] == ["get", "pv"]:
        if state.get("fail_pv_lookup"):
            code, error = 1, "Forbidden: cannot list persistentvolumes"
        else:
            output = "".join(pv + " " + ns + "\n" for pv, ns in state.get("pvs", []))
    elif args[:2] == ["delete", "namespace"]:
        state["namespace"] = False
    elif args[:2] == ["wait", "--for=delete"]:
        if state.get("wait_fails"):
            code, error = 1, "timed out waiting for disk reclamation"
        else:
            state["pvs"] = [p for p in state["pvs"] if "pv/" + p[0] != args[2]]
    else:
        code, error = 1, "Unexpected kubectl command: " + str(args)
path.write_text(json.dumps(state))
print(output, end="")
if error:
    print(error, file=sys.stderr)
sys.exit(code)
'''


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        shutil.copytree(SAMPLE / "scripts", self.root / "scripts")
        self.state_file = self.root / "state.json"
        for tool in ("gcloud", "kubectl", "sleep"):
            executable = self.bin / tool
            executable.write_text("#!" + sys.executable + "\n" + COMMAND)
            executable.chmod(0o755)
        for tool in ("dirname", "awk", "mktemp", "cat", "rm", "grep", "git"):
            (self.bin / tool).symlink_to(shutil.which(tool))
        # Exclude the developer's installed plugin, while keeping basic shell tools.
        self.env = dict(os.environ, PATH=str(self.bin),
                        TEST_STATE=str(self.state_file), PROJECT_ID=PROJECT,
                        REGION=REGION, NAME=NAME)
        for variable in ("BUILD_SA", "NODE_SA", "BASH_ENV", "ENV"):
            self.env.pop(variable, None)
        # Keep prerequisite tests independent of the system Python installation.
        python = self.bin / "python3"
        python.symlink_to(sys.executable)
        self.write_state(calls=[])

    def write_state(self, **state):
        state.setdefault("calls", [])
        self.state_file.write_text(json.dumps(state))

    def read_state(self):
        return json.loads(self.state_file.read_text())

    def run_script(self, script="cleanup.sh"):
        return subprocess.run(["/bin/bash", "scripts/" + script], cwd=self.root,
                              env=self.env, capture_output=True, text=True, timeout=15)

    def test_reclaims_all_namespace_volumes_before_cluster_deletion(self):
        self.write_state(cluster=True, repository=True, subnet=True, network=True,
                         namespace=True, pvs=[["disk-a", NAME], ["disk-b", NAME],
                                              ["keep", "other-namespace"]])
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.read_state()
        self.assertEqual(state["pvs"], [["keep", "other-namespace"]])
        calls = state["calls"]
        waits = [i for i, c in enumerate(calls) if c[:2] == ["kubectl", "wait"]]
        deletion = next(i for i, c in enumerate(calls) if c[:4] ==
                        ["gcloud", "container", "clusters", "delete"])
        self.assertEqual(len(waits), 2)
        self.assertLess(max(waits), deletion)
        self.assertTrue(all(not state[k] for k in ("cluster", "repository", "subnet", "network")))

    def test_retry_still_waits_when_namespace_and_claim_are_gone(self):
        self.write_state(cluster=True, namespace=True, pvs=[["pending-disk", NAME]], wait_fails=True)
        for _ in range(2):
            result = self.run_script()
            self.assertNotEqual(result.returncode, 0)
            state = self.read_state()
            self.assertFalse(state["namespace"])
            self.assertTrue(state["cluster"])
            self.assertEqual(state["pvs"], [["pending-disk", NAME]])
        state["wait_fails"] = False
        self.write_state(**state)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.read_state()
        self.assertFalse(state["cluster"])
        self.assertEqual(state["pvs"], [])
        self.assertEqual(sum(c[:2] == ["kubectl", "wait"] for c in state["calls"]), 3)

    def test_resource_lookup_errors_are_visible_and_stop_cleanup(self):
        for kind in ("cluster", "repository", "subnet", "network"):
            for error in ("PERMISSION_DENIED", "UNAVAILABLE: temporary service failure"):
                with self.subTest(kind=kind, error=error):
                    self.write_state(fail_lookup=kind, lookup_error=error)
                    result = self.run_script()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertFalse(any("delete" in c for c in self.read_state()["calls"]))

    def test_pv_lookup_failure_preserves_namespace_and_cluster(self):
        self.write_state(cluster=True, namespace=True, fail_pv_lookup=True)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Forbidden", result.stderr)
        state = self.read_state()
        self.assertTrue(state["namespace"] and state["cluster"])

    def test_absent_resources_are_a_successful_noop(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.read_state()["calls"]
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("list" in c and "--project=" + PROJECT in c for c in calls))
        self.assertIn("--location=" + REGION, calls[0])
        self.assertIn("--filter=name=" + NAME, calls[0])
        self.assertIn("--filter=name=projects/" + PROJECT + "/locations/" + REGION +
                      "/repositories/" + NAME, calls[1])
        self.assertIn("--regions=" + REGION, calls[2])
        self.assertIn("--filter=name=" + NAME, calls[2])
        self.assertIn("--filter=name=" + NAME, calls[3])

    def test_cluster_deletion_retries_an_active_operation(self):
        self.write_state(cluster=True, delete_busy=True)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.read_state()["calls"]
        self.assertEqual(sum(c[:4] == ["gcloud", "container", "clusters", "delete"] for c in calls), 2)

    def test_missing_auth_plugin_stops_before_cloud_calls(self):
        result = self.run_script("prepare.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required command: gke-gcloud-auth-plugin", result.stderr)
        self.assertIn("#install_plugin", result.stderr)
        self.assertEqual(self.read_state()["calls"], [])


if __name__ == "__main__":
    unittest.main()
