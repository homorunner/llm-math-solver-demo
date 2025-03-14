import os
import subprocess
import tempfile
from multiprocessing import Pool


class PythonExecutor:
    def __init__(self, timeout=5, concurrency=0):
        self.timeout = timeout
        self.concurrency = concurrency if concurrency > 0 else os.cpu_count()

    def run(self, query):
        if "print(" not in query:
            return "", "No print statement found in the code"
        for x in ["input()", "readline()", "stdin.read()"]:
            if x in query:
                return "", x+" is not allowed, use constants instead"
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
                output_lines = output.split("\n")
                if len(output_lines) > 3:
                    output = "\n".join(output_lines[-3:])
                    for i in range(len(output_lines) - 3):
                        if len(output_lines[i]) > 102:
                            output_lines[i] = output_lines[i][:100] + "..."
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
            if len(new_msgs) > 7:
                new_msgs = new_msgs[-7:]
            for i in range(len(new_msgs)):
                if len(new_msgs[i]) > 102:
                    new_msgs[i] = new_msgs[i][:100] + "..."
            error_msg = "\n".join(new_msgs)
            return "", error_msg.strip()

    def safe_run(self, query):
        try:
            res, report = self.run(query)
            return res, report
        except (TimeoutError, subprocess.TimeoutExpired):
            return "", "Execution time out"
        except Exception as e:
            return "", str(e)

    def batch_apply(self, queries):
        with Pool(processes=self.concurrency) as pool:
            results = pool.map(self.safe_run, queries)
        return results
