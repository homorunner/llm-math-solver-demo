import os
import subprocess
import tempfile


class PythonExecutor:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def run(self, query):
        if "print(" not in query:
            return "", "No print statement found in the code"
        if "input(" in query:
            return "", "input() is not allowed, use constants instead"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            result = subprocess.run(
                ["python3", temp_file_path],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                output = result.stdout
                return output.strip(), "Done"
            error_msg = result.stderr.strip()
            msgs = error_msg.split("\n")
            new_msgs = []
            want_next = False
            for m in msgs:
                if m == msgs[-1]:
                    new_msgs.append(m)
                elif temp_file_path in m:
                    want_next = True
                elif want_next:
                    new_msgs.append(m.strip())
                    want_next = False
            error_msg = "\n".join(new_msgs)
            return "", error_msg.strip()

    def batch_apply(self, queries):
        results = []
        for query in queries:
            try:
                res, report = self.run(query)
                results.append((res, report))
            except (TimeoutError, subprocess.TimeoutExpired) as e:
                results.append(("", "Execution time out"))
            except Exception as e:
                results.append(("", str(e)))
        return results
