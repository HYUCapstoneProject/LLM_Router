from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)

from ml.kubeflow.components.modernbert_training import (
    modernbert_continual_training_component,
    modernbert_training_component,
)


VERTEX_TRAINING_MACHINE_TYPE = "n1-standard-4"
VERTEX_TRAINING_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
VERTEX_TRAINING_ACCELERATOR_COUNT = 1
VERTEX_TRAINING_TIMEOUT = "21600s"
VERTEX_TRAINING_BOOT_DISK_TYPE = "pd-ssd"
VERTEX_TRAINING_BOOT_DISK_SIZE_GB = 100


vertex_modernbert_training_job_component = create_custom_training_job_from_component(
    component_spec=modernbert_training_component,
    display_name="ModernBERT router T4 training",
    replica_count=1,
    machine_type=VERTEX_TRAINING_MACHINE_TYPE,
    accelerator_type=VERTEX_TRAINING_ACCELERATOR_TYPE,
    accelerator_count=VERTEX_TRAINING_ACCELERATOR_COUNT,
    boot_disk_type=VERTEX_TRAINING_BOOT_DISK_TYPE,
    boot_disk_size_gb=VERTEX_TRAINING_BOOT_DISK_SIZE_GB,
    timeout=VERTEX_TRAINING_TIMEOUT,
    restart_job_on_worker_restart=False,
    strategy="STANDARD",
)


vertex_modernbert_continual_training_job_component = (
    create_custom_training_job_from_component(
        component_spec=modernbert_continual_training_component,
        display_name="ModernBERT router continual T4 training",
        replica_count=1,
        machine_type=VERTEX_TRAINING_MACHINE_TYPE,
        accelerator_type=VERTEX_TRAINING_ACCELERATOR_TYPE,
        accelerator_count=VERTEX_TRAINING_ACCELERATOR_COUNT,
        boot_disk_type=VERTEX_TRAINING_BOOT_DISK_TYPE,
        boot_disk_size_gb=VERTEX_TRAINING_BOOT_DISK_SIZE_GB,
        timeout=VERTEX_TRAINING_TIMEOUT,
        restart_job_on_worker_restart=False,
        strategy="STANDARD",
    )
)
