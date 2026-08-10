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


def configure_gpu_training_task(task: dsl.PipelineTask, *, image: str) -> None:
    """Apply the shared single-GPU training policy to initial and retraining jobs."""
    task.set_container_image(image)
    task.set_accelerator_type("nvidia.com/gpu")
    task.set_accelerator_limit(1)
    task.set_cpu_request("4")
    task.set_cpu_limit("8")
    task.set_memory_request("24Gi")
    task.set_memory_limit("32Gi")
    task.set_retry(
        num_retries=2,
        backoff_duration="60s",
        backoff_factor=2.0,
        backoff_max_duration="600s",
    )
    task.set_caching_options(enable_caching=True)


def configure_gpu_evaluation_task(task: dsl.PipelineTask, *, image: str) -> None:
    """Apply a smaller single-GPU policy to full RouterArena inference."""
    task.set_container_image(image)
    task.set_accelerator_type("nvidia.com/gpu")
    task.set_accelerator_limit(1)
    task.set_cpu_request("2")
    task.set_cpu_limit("4")
    task.set_memory_request("8Gi")
    task.set_memory_limit("16Gi")
    task.set_retry(
        num_retries=2,
        backoff_duration="60s",
        backoff_factor=2.0,
        backoff_max_duration="600s",
    )
    task.set_caching_options(enable_caching=True)
