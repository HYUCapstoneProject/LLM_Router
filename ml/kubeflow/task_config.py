from kfp import dsl


def configure_cpu_task(
    task: dsl.PipelineTask,
    *,
    image: str,
    cpu_request: str,
    cpu_limit: str,
    memory_request: str,
    memory_limit: str,
) -> None:
    """Apply the shared CPU task policy used by data-processing components."""
    task.set_container_image(image)
    task.set_cpu_request(cpu_request)
    task.set_cpu_limit(cpu_limit)
    task.set_memory_request(memory_request)
    task.set_memory_limit(memory_limit)
    task.set_retry(
        num_retries=2,
        backoff_duration="30s",
        backoff_factor=2.0,
        backoff_max_duration="300s",
    )
    task.set_caching_options(enable_caching=True)
