"""The AWS run as one SageMaker Pipeline (README, "Running on AWS").

    python -m enefit.pipeline --image <ECR image URI> [--dry-run]

Steps, each starting only when the one before it succeeded:
  features   processing job (ml.m5.xlarge): all eight feature tables -> s3://<bucket>/data/features
  v1 .. v6   one training job each (ml.m6i.2xlarge): registry entries #1-#6, v6 becomes @champion
  live-run   training job (ml.m6i.2xlarge): enefit.live_run
The steps hand data over through S3: every training job mounts s3://<bucket>/data read-only and logs to the MLflow app.
Creates the pipeline "enefit" (or replaces its definition), then starts an execution named after the image tag.
"""
from __future__ import annotations

import argparse
import json

import boto3

REGION = "us-east-1"
NAME = "enefit"
ROLE_NAME = "enefit-sagemaker"
MILESTONES = ["v1", "v2", "v3", "v4", "v5", "v6"]
WORKERS = ["--workers", "4", "--threads", "2"]   # ml.m6i.2xlarge, docs/aws_sizing.md


def definition(image: str, role: str, bucket: str, mlflow: str) -> dict:
    data = "/opt/ml/processing/data"
    s3_input = lambda name: {"InputName": name, "S3Input": {
        "S3Uri": f"s3://{bucket}/data/{name}", "LocalPath": f"{data}/{name}", "S3DataType": "S3Prefix", "S3InputMode": "File"}}
    steps = [{"Name": "features", "Type": "Processing", "Arguments": {
        "RoleArn": role,
        "AppSpecification": {"ImageUri": image,
                             "ContainerEntrypoint": ["python", "-m", "enefit.features", "--build", "--feature-config", "all"]},
        "Environment": {"ENEFIT_DATA_DIR": data},
        "ProcessingInputs": [s3_input("raw"), s3_input("ext")],
        "ProcessingOutputConfig": {"Outputs": [{"OutputName": "features", "S3Output": {
            "S3Uri": f"s3://{bucket}/data/features", "LocalPath": f"{data}/features", "S3UploadMode": "EndOfJob"}}]},
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 30}},
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600}}}]

    jobs = [(v, ["-m", "enefit.train", "--variant", v, "--register"] + (["--champion"] if v == "v6" else []))
            for v in MILESTONES] + [("live-run", ["-m", "enefit.live_run"])]
    for name, args in jobs:
        steps.append({"Name": name, "Type": "Training", "DependsOn": [steps[-1]["Name"]], "Arguments": {
            "RoleArn": role,
            "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "FastFile",
                                       "ContainerEntrypoint": ["python"], "ContainerArguments": args + WORKERS},
            "InputDataConfig": [{"ChannelName": "data", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/data", "S3DataDistributionType": "FullyReplicated"}}}],
            "Environment": {"ENEFIT_DATA_DIR": "/opt/ml/input/data/data", "MLFLOW_TRACKING_URI": mlflow},
            "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/jobs"},
            "ResourceConfig": {"InstanceType": "ml.m6i.2xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30},
            "StoppingCondition": {"MaxRuntimeInSeconds": 10800}}})
    return {"Version": "2020-12-01", "Steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="ECR image URI, tagged with its commit (README, B)")
    ap.add_argument("--dry-run", action="store_true", help="print the pipeline definition; create and start nothing")
    a = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    account = session.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/{ROLE_NAME}"
    mlflow = next(app["Arn"] for app in sm.list_mlflow_apps()["Summaries"] if app["Name"] == NAME)
    body = definition(a.image, role, f"enefit-mlops-{account}", mlflow)
    if a.dry_run:
        print(json.dumps(body, indent=1))
        return

    exists = any(p["PipelineName"] == NAME for p in sm.list_pipelines(PipelineNamePrefix=NAME)["PipelineSummaries"])
    (sm.update_pipeline if exists else sm.create_pipeline)(PipelineName=NAME, PipelineDefinition=json.dumps(body), RoleArn=role)
    tag = a.image.rsplit(":", 1)[-1]
    arn = sm.start_pipeline_execution(PipelineName=NAME, PipelineExecutionDisplayName=f"image-{tag}",
                                      PipelineExecutionDescription=a.image)["PipelineExecutionArn"]
    print(f"{'updated' if exists else 'created'} pipeline {NAME}; started {arn}")


if __name__ == "__main__":
    main()
