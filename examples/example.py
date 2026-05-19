import time
import random
from concurrent.futures import ThreadPoolExecutor
from multitqdm import TaskProgressBar, ProgressBarExecutor

def worker(task: TaskProgressBar, steps: int) -> None:
    """Example worker that reports progress without touching tqdm."""

    with task.start(desc=f"Task {task.task_id}", total=steps) as pb:
        for _ in range(steps):
            time.sleep(random.uniform(0.02, 0.12))
            pb.progress(1)


def main() -> None:
    tasks = [40, 55, 35, 70, 40, 55, 35, 70, 40, 55, 35, 70]
    worker_count = 4

    with ProgressBarExecutor(
            ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="worker",
            ),
            desc="Total",
            total=len(tasks),
            total_completed=True
    ) as executor:
        for future in [executor.submit(worker, steps) for steps in tasks]:
            future.result()


if __name__ == "__main__":
    main()