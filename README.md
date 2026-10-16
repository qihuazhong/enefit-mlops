# enefit-mlops

The Enefit Kaggle model ("Predict Energy Behavior of Prosumers", built Nov 2023 – Jan 2024),
productionized in October 2026: one feature path for training and scoring, MLflow tracking and registry,
evaluation on a fixed time split, drift monitoring, and online retraining.

## Data split (fixed)
| Block  | Dates                    | Use |
|--------|--------------------------|-----|
| train  | 2021-09-01 → 2023-02-28  | fitting; forward-chaining CV inside for early stopping |
| test   | 2023-03-01 → 2023-04-15  | reported for every version |
| live run   | 2023-04-16 → 2023-05-31  | held-out days, one at a time as if live: daily scoring, drift checks, weekly retraining |

## Layout
- `src/enefit/`: `config` (all settings), `data` (raw tables, availability rules), `weather`, `features` (one builder for
  training and scoring), `targets` (derived targets, ensemble reconstruction), `train`, `model` (MLflow pyfunc), `evaluate`,
  `live_run` (the held-out days, one at a time, as if live), `pipeline` (the AWS run as one SageMaker Pipeline)
- `tests/`: training/serving parity, no look-ahead, reconstruction, packaged model, one test per incident
- `docs/milestone_variants.md` (how the notebook versions differ), `docs/benchmarks.md` (comparison against
  SARIMAX, Prophet, Croston/TSB and DeepAR)
- `reference/`: original notebook versions and the Kaggle submission log

Data (`data/raw`, `data/ext`) is not committed: the Kaggle competition files and the county pixel weights.

## What broke when moving the notebook to production
The Kaggle notebook scored well, but a few things in it would have failed or misled once the model ran every day.
Each one has a regression test in `tests/test_porting_fixes.py`.

- **Weather forecasts from the future.** For each hour the notebook averaged every forecast issued for it,
  including one issued on the day being predicted, which is not available yet at prediction time. Evaluated that
  way the model looked about 3 MAE better than it does on the forecasts it would really have (47.6 vs 50.5 on the
  test block). Training still uses the averaged forecasts (it helps); evaluation and scoring use only the
  forecasts available at prediction time.
- **Daylight saving time.** A few hours are missing at the spring change, and lags were taken by row count, so
  some lags were off by one hour. Lags are now taken on a complete hourly grid.
- **Lags leaking across segments.** Shifts ran on the concatenated table, so the first days of one segment read
  the end of another. Shifts now run within each segment.
- **Rolling statistics that depend on how much history is loaded.** pandas computes rolling skew and kurtosis
  with running sums, so a shorter scoring history gave different values than training. Each window is now
  computed on its own.
- **Validation that peeked at the future.** The notebook grouped folds by day of the month, which puts later days
  in the training folds. Cross-validation now only trains on the past.
- **A hang on AWS.** The live run worked on my Mac but sat idle on SageMaker until its 3-hour limit: on Linux,
  worker processes are forked from a parent that has already run LightGBM, and they wait forever. Workers now
  start with "spawn".

## Reproducing the runs
Run everything from the repo root. Activate the environment once per terminal; then every step is `python -m enefit.<step>`
(`--help` on any step lists its options).

**1. Setup (once)**
```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[dev,benchmarks]"   # benchmarks: torch, prophet, statsmodels (enefit.benchmarks)
source .venv/bin/activate
```

**2. Data (once).** The Kaggle competition files (accept the competition rules on kaggle.com first) and the county pixel weights:
```bash
kaggle competitions download -c predict-energy-behavior-of-prosumers -p data/raw
unzip -o data/raw/predict-energy-behavior-of-prosumers.zip -d data/raw
kaggle datasets download qihuaz/enefit-county-coordinate-weights -p data/ext/county-weights --unzip
```

**3. Feature tables** (about 20 s each). `serving` uses the forecasts available at prediction time; `train` averages every
forecast issued for an hour (see "What broke" below). Tables for older milestones are built automatically the first time they are needed
(`--feature-config all` builds all eight). Each table records the hash of the input files and of the feature code; training
rebuilds a table that no longer matches either.
```bash
python -m enefit.features --build
python -m enefit.features --build --feature-config train
```

**4. MLflow** (tracking + model registry), in its own terminal, left running. Open http://127.0.0.1:5050
(not port 5000: macOS AirPlay Receiver also listens there).
```bash
source .venv/bin/activate
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5050
```

**5. The six notebook milestones**, one run each, in this order (about 1–7 min each). They become registry entries #1–#6;
v6 is made the live model (`@champion`). v1–v6 are numbered in order; each comes from a saved version of the Kaggle notebook:

| Model | Kaggle notebook version | Date | Main change |
|---|---|---|---|
| v1 | 84 | 2023-11-08 | one LightGBM model per side |
| v2 | 125 | 2023-11-15 | 7 targets per side (target + differences to past values), weighted weather grid |
| v3 | 158 | 2023-12-31 | data augmentation for capacity growth |
| v4 | 190 | 2024-01-05 | weather bias correction |
| v5 | 217 | 2024-01-09 | counterpart-lag bug fixed, more augmentation |
| v6 | 238 | 2024-01-15 | final training notebook (float16, tuned regularisation) |

All six run with v6's float16 feature cast (v1–v5's notebooks used float64; the `notebook_fidelity` tag says so): a
float64 feature table needs about 16 GiB to build, float16 about 6 GiB.

```bash
python -m enefit.train --variant v1 --register
python -m enefit.train --variant v2 --register
python -m enefit.train --variant v3 --register
python -m enefit.train --variant v4 --register
python -m enefit.train --variant v5 --register
python -m enefit.train --variant v6 --register --champion
```
Each run trains on the train block with the same recipe, scores the test block and logs to MLflow as `milestone-<variant>`.
Older milestones whose notebook used something the feature builder does not have (nearest-grid weather, EET shift, bias
correction, client join on `data_block_id`, per-county snowfall sum) run on the current builder; the `notebook_fidelity`
tag on the registered model says what is approximated.

**6. Live run** (about 13 min). Walks through 2023-04-16 → 05-31 one day at a time as if deployed: each day it
predicts with the live model, checks accuracy when the actual values arrive 2 days later and runs the drift checks; each
Sunday it retrains v6 on all data so far and runs the retrained model next to the live one for a week. It then goes live
only if it passes the gate (`live_run.gate`):
- **non-inferiority**: with 90 % confidence (bootstrap over days), its total error is less than 2 % above the live model's;
- **guardrails**: neither side more than 2 % worse, neither side's bias more than 2 points worse, valid predictions.

The model it replaced keeps scoring for one more week; if the new model then fails the same gate against it, the
`@champion` alias is moved back (**rollback**). In MLflow: one run "live run …" (the daily log) with the 5
retrains listed under it; each retrain carries its decision and the reason.
```bash
python -m enefit.live_run
```
It starts from the newest v6 of step 5 (not a retrain), so a re-run starts from the same model; another start
is `--start-champion <registry version>`. Outputs (daily log, decisions, chart) go to `outputs/live_run/<timestamp>/`.

**7. Tests**
```bash
pytest -q
```

## Running on AWS (SageMaker, us-east-1)
The same code in one Docker image, run as SageMaker jobs chained in one SageMaker Pipeline. S3 holds the data, the feature tables and the MLflow artifacts.
MLflow is a SageMaker serverless MLflow app (tracking + registry; no server to run).

| Step | Runs as | Instance | Reads | Writes |
|---|---|---|---|---|
| feature tables | processing job | ml.m5.xlarge | `s3://…/data/raw`, `data/ext` | `s3://…/data/features` |
| milestones v1–v6 | one training job each | ml.m6i.2xlarge, 4 workers × 2 threads | `s3://…/data` (read-only mount) | MLflow |
| live run | training job | ml.m6i.2xlarge, 4 workers × 2 threads | `s3://…/data` (read-only mount) | MLflow |

Why these instances: `docs/aws_sizing.md`. Commands need the AWS CLI v2, Docker, and a signed-in session (`aws login`).

**Settings for every command below** (one terminal):
```bash
export AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=enefit-mlops-$ACCOUNT
ROLE=arn:aws:iam::${ACCOUNT}:role/enefit-sagemaker   # braces: in zsh, $ACCOUNT:r would mean "remove extension"
IMAGE=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/enefit:$(git rev-parse --short HEAD)   # the commit checked out now
MLFLOW=$(aws sagemaker list-mlflow-apps --query "Summaries[?Name=='enefit'].Arn" --output text)   # empty until A has run
```

**A. Once: budget alert, bucket and data, role, MLflow app.** The budget emails when the month's spend passes $20.
The role is used by the jobs (S3, ECR, logs, MLflow) and by the pipeline (starts the jobs and passes them the role).
```bash
aws budgets create-budget --account-id $ACCOUNT --budget '{"BudgetName":"enefit","BudgetLimit":{"Amount":"20","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":100,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"YOUR_EMAIL"}]}]'
aws s3 mb s3://$BUCKET
aws s3 sync data/raw s3://$BUCKET/data/raw --exclude "*" --include "train.csv" --include "client.csv" --include "electricity_prices.csv" --include "forecast_weather.csv" --include "historical_weather.csv" --include "weather_station_to_county_mapping.csv" --include "example_test_files/*"
aws s3 cp data/ext/county-weights/county_pixel_weights.csv s3://$BUCKET/data/ext/county-weights/
aws iam create-role --role-name enefit-sagemaker --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name enefit-sagemaker --policy-name enefit --policy-document '{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:GetBucketLocation"],"Resource":["arn:aws:s3:::'$BUCKET'","arn:aws:s3:::'$BUCKET'/*"]},
  {"Effect":"Allow","Action":["ecr:GetAuthorizationToken","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer","ecr:BatchCheckLayerAvailability"],"Resource":"*"},
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams","cloudwatch:PutMetricData"],"Resource":"*"},
  {"Effect":"Allow","Action":["sagemaker-mlflow:*","sagemaker:CallMlflowAppApi"],"Resource":"*"},
  {"Effect":"Allow","Action":["sagemaker:CreateProcessingJob","sagemaker:DescribeProcessingJob","sagemaker:StopProcessingJob","sagemaker:CreateTrainingJob","sagemaker:DescribeTrainingJob","sagemaker:StopTrainingJob","sagemaker:AddTags"],"Resource":"*"},
  {"Effect":"Allow","Action":"iam:PassRole","Resource":"'$ROLE'","Condition":{"StringEquals":{"iam:PassedToService":"sagemaker.amazonaws.com"}}}]}'
aws sagemaker create-mlflow-app --name enefit --artifact-store-uri s3://$BUCKET/mlflow --role-arn $ROLE
```
The app takes a few minutes to start. Then set `MLFLOW` again (from the settings) and check it is ready and runs the
MLflow version `pyproject.toml` pins (prints `Created 3.10.1`):
```bash
aws sagemaker describe-mlflow-app --arn $MLFLOW --query '[Status,MlflowVersion]' --output text
```

**GitHub Actions access (once).** The workflows sign in to AWS with GitHub's OIDC token: AWS trusts GitHub as an
identity provider, and a workflow on `main` of this repo (and nothing else) may use the role `enefit-github`, which can
push images, read the raw data, and start the pipeline. No AWS keys are stored in GitHub. The `sub` the role accepts is
this repo's subject claim, with owner and repo ids (`gh api repos/qihuazhong/enefit-mlops/actions/oidc/customization/sub`
prints its prefix), plus `:ref:refs/heads/main`.
```bash
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com
aws iam create-role --role-name enefit-github --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Federated":"arn:aws:iam::'${ACCOUNT}':oidc-provider/token.actions.githubusercontent.com"},"Action":"sts:AssumeRoleWithWebIdentity",
  "Condition":{"StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com","token.actions.githubusercontent.com:sub":"repo:qihuazhong@22231207/enefit-mlops@1403798946:ref:refs/heads/main"}}}]}'
aws iam put-role-policy --role-name enefit-github --policy-name ci --policy-document '{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
  {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:DescribeImages"],"Resource":"arn:aws:ecr:'$AWS_REGION':'${ACCOUNT}':repository/enefit"},
  {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::'$BUCKET'"},
  {"Effect":"Allow","Action":"s3:GetObject","Resource":["arn:aws:s3:::'$BUCKET'/data/raw/*","arn:aws:s3:::'$BUCKET'/data/ext/*"]},
  {"Effect":"Allow","Action":["sagemaker:ListMlflowApps","sagemaker:ListPipelines","sagemaker:CreatePipeline","sagemaker:UpdatePipeline","sagemaker:StartPipelineExecution"],"Resource":"*"},
  {"Effect":"Allow","Action":"iam:PassRole","Resource":"'$ROLE'","Condition":{"StringEquals":{"iam:PassedToService":"sagemaker.amazonaws.com"}}}]}'
gh variable set AWS_ROLE_ARN --body arn:aws:iam::${ACCOUNT}:role/enefit-github   # the workflows read the role from this repo variable
```

**B. Image.** GitHub Actions builds it (`.github/workflows/image.yml`) on every push to `main` that changes the code:
build from the committed tree, run all tests inside the image (with the serving table built in the container, on x86),
push to ECR tagged with the commit. A commit already in ECR is not rebuilt. The same by hand:
```bash
aws ecr create-repository --repository-name enefit --image-tag-mutability IMMUTABLE
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
IMAGE=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/enefit:$(git rev-parse --short HEAD)
git archive HEAD | docker build --platform linux/amd64 --provenance=false --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t $IMAGE -
docker run --rm --platform linux/amd64 -v "$PWD/data:/data:ro" -e ENEFIT_DATA_DIR=/data $IMAGE pytest -q --deselect tests/test_parity.py::test_scoring_path_equals_full_parquet
docker push $IMAGE
```
Notes:
- `create-repository` is needed only the first time. Tags are immutable: a pushed tag always means that commit.
- `IMAGE` is set again so the tag is the commit being built, even if you committed since the settings.
- `--provenance=false`: push a plain image, not an image index with a build attestation (which SageMaker may not accept).
- The test run mounts the data read-only, so it also checks that nothing writes into it.
- The skipped test compares against the tables on disk, which were built on the Mac and differ from x86 in the last bit.

**C. Run the workflow** (a SageMaker Pipeline, `src/enefit/pipeline.py`). It creates the pipeline (or updates it)
and starts it; each step starts when the one before it has succeeded:
features (processing job) → v1 → v2 → … → v6 (`@champion`) → live run (training jobs).
From GitHub: Actions → run-pipeline → Run workflow (optionally an image tag; default: the latest commit on `main`, whose
image must exist). From a terminal:
```bash
python -m enefit.pipeline --image $IMAGE
```
- `--dry-run` prints the pipeline definition (the job settings: image, instance, arguments) and starts nothing.
- Watch it: `aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn <arn> --query 'PipelineExecutionSteps[].[StepName,StepStatus]' --output table`
  (the ARN is printed when it starts). Job logs are in CloudWatch, under
  `/aws/sagemaker/ProcessingJobs` and `/aws/sagemaker/TrainingJobs`.
- If a step fails: fix the cause, then `aws sagemaker retry-pipeline-execution --pipeline-execution-arn <arn>` reruns
  that step and the ones after it (the steps that succeeded are not rerun). A new execution also works: it
  registers v1–v6 again (#7–#12) and the live run starts from that v6.

**MLflow UI** (a link valid for a few minutes): `aws sagemaker create-presigned-mlflow-app-url --arn $MLFLOW`.

**Teardown.** Jobs stop on their own and cost nothing afterwards. The pipeline and the MLflow app have no hourly charge; S3 and ECR cost
cents per month. To remove everything:
```bash
aws sagemaker delete-pipeline --pipeline-name enefit
aws sagemaker delete-mlflow-app --arn $MLFLOW
aws ecr delete-repository --repository-name enefit --force
aws s3 rb s3://$BUCKET --force
aws iam delete-role-policy --role-name enefit-sagemaker --policy-name enefit
aws iam delete-role --role-name enefit-sagemaker
aws iam delete-role-policy --role-name enefit-github --policy-name ci
aws iam delete-role --role-name enefit-github
aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com
aws budgets delete-budget --account-id $ACCOUNT --budget-name enefit
```

## How it differs from the notebook (v6)
- **Features**: ported from v6; the porting fixes below are always on. Availability rules
  (`data.available_at`, checked against the Kaggle API example files): targets and client data of D-2, prices of D-1,
  forecasts issued by D-1, historical weather through lags of 48 h or more.
- **Recipe**: training rows use forecasts averaged over every origin; CV validation, test and serving use the origins
  available at prediction time.
- **CV**: forward-chaining (validation 2022-05/06, 2022-09/10, 2023-01/02; 2-day embargo) instead of v6's GroupKFold by day of
  month, which mixed future days into training folds.
- **Model**: one refit booster per (side, target) with the median fold best iteration, instead of averaging fold boosters.
  Augmentation is applied to training rows only. Each side is one MLflow pyfunc returning the final reconstructed prediction.
- **Evaluation**: raw float64 target of train.csv; naive baseline = same hour 7 days earlier. MAE (kWh per segment-hour) drives model
  selection; WAPE (share of the energy missed) and relative bias (share over/under-forecast) are logged next to it for
  non-technical readers.
- **Baseline from the notebook's saved outputs** (v6 `lgb.cv`, optimistic CV): production `target` 24.85, consumption 44.88.
