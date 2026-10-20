"""Exercise the launcher without activating conda or starting GPU training."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "baselines/mc/train.sh"


class LauncherTests(unittest.TestCase):
    def run_launcher(self, *, exit_code=0, resume="", launcher=LAUNCHER, settings=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "baselines/mc/train.sh"
            script.parent.mkdir(parents=True)
            binary = root / "bin"
            binary.mkdir()
            python = binary / "python"
            python.write_text(
                "#!/bin/bash\n"
                'printf "gpu=%s jax=%s\\n" "$CUDA_VISIBLE_DEVICES" "$JAX_PLATFORMS"\n'
                'printf "%s\\n" "$@"\n'
                'echo "mock stderr" >&2\n'
                'exit "$MOCK_EXIT"\n'
            )
            python.chmod(0o755)
            conda = root / "conda.sh"
            conda.write_text(
                'conda() { [[ "$*" == "activate test-env" ]]; }\n'
                f'export PATH={shlex.quote(str(binary))}:"$PATH"\n'
            )
            source = launcher.read_text()
            fixture = {
                "CONDA_SH": shlex.quote(str(conda)),
                "CONDA_ENV": '"test-env"',
                "CUDA_VISIBLE_DEVICES": "2",
                "SEEDS": "(1)" if resume else "(1 2)",
                "ITERATIONS": "1000",
                "NUM_ENVS": "16",
                "RESUME": shlex.quote(resume),
            } | (settings or {})
            for name, value in fixture.items():
                source = re.sub(
                    rf"^((?:export )?{name}=).*$",
                    lambda match, value=value: match[1] + value,
                    source,
                    flags=re.MULTILINE,
                )
            script.write_text(source)
            result = subprocess.run(
                ["bash", str(script)],
                check=False,
                cwd=root,
                env=os.environ | {"MOCK_EXIT": str(exit_code)},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
            )
            outputs = [p.read_text() for p in sorted(root.glob("logs/*/*.out"))]
            logs = [p.read_text() for p in sorted(root.glob("logs/*/*.log"))]
            return result, outputs, logs

    def test_defaults_and_sequential_seed_logs(self):
        result, outputs, logs = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(outputs), 2)
        for seed, output, log in zip((1, 2), outputs, logs):
            self.assertIn("gpu=2 jax=cpu", output)
            self.assertIn(f"--seed\n{seed}\n", output)
            self.assertIn("--iterations\n1000\n", output)
            self.assertIn("--num-envs\n16\n", output)
            self.assertIn("--device\ncuda\n", output)
            self.assertIn("mock stderr", log)
            self.assertIn("PID:", log)
            self.assertIn("Completed job:", log)
        self.assertLess(
            result.stdout.index("Completed job:"),
            result.stdout.rindex("Starting job:"),
        )

    def test_failure_preserves_exit_code_and_stops_next_seed(self):
        result, outputs, logs = self.run_launcher(exit_code=7)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(outputs), 1)
        self.assertIn("Failed job:", logs[0])
        self.assertNotIn("Completed job:", logs[0])

    def test_resume_reuses_checkpoint_directory(self):
        checkpoint = ".local_training/mc/existing/last.pt"
        result, outputs, _ = self.run_launcher(resume=checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(outputs), 1)
        self.assertIn(f"--resume\n{checkpoint}\n", outputs[0])
        self.assertIn("--output\n.local_training/mc/existing\n", outputs[0])

    def test_ismcts_launcher_selects_gpu_and_passes_search_settings(self):
        result, outputs, logs = self.run_launcher(
            launcher=LAUNCHER.parents[1] / "ismcts/run.sh",
            settings={"ITERATIONS": "1024", "SEEDS": "(2000000)"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(outputs), 1)
        self.assertIn("gpu=2 jax=cuda,cpu", outputs[0])
        self.assertIn("--iterations\n1024\n", outputs[0])
        self.assertIn("--device\ngpu\n", outputs[0])
        self.assertIn("Completed job:", logs[0])


if __name__ == "__main__":
    unittest.main()
