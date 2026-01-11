## Project Overview: Hybrid Reddit Data Pipeline with Automated CI/CD

This project demonstrates the transition from a manual, local-only development workflow to a streamlined **Hybrid-Cloud** architecture. The goal was to eliminate manual redundancies, improve deployment reliability, and make data insights publicly accessible without incurring high cloud compute costs.

### 1. The Operational Challenge (The "Before" State)

Initially, the project relied on a standard local development setup, which created three specific inefficiencies:

* **Manual Dependency:** Every code change required manually stopping containers, rebuilding Docker images, and restarting services. This was time-consuming and prone to human error.
* **Environment Inconsistency:** The application ran strictly on `localhost`. Sharing insights meant sending screenshots or manually transferring data, creating a "works on my machine" silo.
* **Lack of Version Control in Artifacts:** Without a container registry, there was no easy way to roll back to a previous stable version of the build if the local build failed.

### 2. The Solution: A Decoupled CI/CD Architecture

To address these issues, I separated the architecture into three distinct components: **Compute** (Local), **Storage** (Cloud), and **Presentation** (Cloud), tied together by an automated pipeline.

#### The Architecture Breakdown

| Component | Service | Role |
| --- | --- | --- |
| **Orchestration** | **GitHub Actions** | Automates code linting, building, and pushing Docker images. |
| **Artifact Registry** | **Docker Hub** | Acts as the "source of truth" for the application image, ensuring consistency across environments. |
| **Backend (Compute)** | **Local Server (Docker)** | Runs the heavy ETL/ML workloads locally to save on cloud compute costs. Pulls verified images from Docker Hub. |
| **Storage Layer** | **AWS S3** | Serves as the data lake, decoupling the backend from the frontend. |
| **Frontend** | **Streamlit Cloud** | Reads directly from S3 to visualize data, allowing for public access without exposing the local network. |

---

### 3. The DevOps Implementation

#### Phase 1: Automating the Build (CI)

Building the Docker image locally was resource-intensive. I moved this process to the cloud using **GitHub Actions**.

* **The Trigger:** Pushing to the `main` branch initiates the workflow.
* **The Action:** The pipeline uses `docker/setup-buildx-action` to build a multi-architecture image (x86/ARM) and securely authenticates with Docker Hub using repository secrets.
* **The Result:** A verified, production-ready image is pushed to `sayuj739/reddit-airflow-pipeline:latest` automatically.

#### Phase 2: Centralizing Artifacts

By using **Docker Hub**, I decoupled the code from the runtime. My local server no longer builds code; it simply pulls the latest stable artifact. This ensures that the code running in production is exactly the same as the code verified in the CI stage.

#### Phase 3: Secure Deployment (CD)

* **Backend:** A simple `docker-compose pull` on the local server updates the pipeline.
* **Frontend:** Streamlit Cloud is connected directly to the GitHub repository. It detects changes in the dashboard code and re-deploys automatically.
* **Security:** AWS credentials are managed via Streamlit Secrets and GitHub Secrets, ensuring no sensitive keys are ever committed to the codebase.

---

### 4. Technical Configuration

**Repository Structure**

```text
RedditDataPipeline/
├── .github/
│   └── workflows/
│       └── docker-publish.yml   # CI Pipeline definition
├── dags/                        # Airflow orchestration logic
├── streamlit_dashboard.py       # Frontend visualization
├── docker-compose.yml           # Runtime configuration
└── Dockerfile                   # Environment definition

```

**CI Pipeline Snippet (`docker-publish.yml`)**
This configuration handles the authentication and push process:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ "main" ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: sayuj739/reddit-airflow-pipeline:latest

```

---

### 5. Summary of Improvements

> **Note on Cost vs. Performance:**
> By keeping the heavy data processing (Airflow/ETL) on a local server and only pushing lightweight CSV/Parquet data to S3, I avoided the high costs of running persistent heavy-compute instances in the cloud, while still maintaining a public-facing cloud dashboard.

**Key Outcomes:**

* **Reliability:** Eliminated "it works on my machine" issues by standardizing on Docker images.
* **Speed:** Reduced deployment time from ~15 minutes (manual build) to ~1 minute (pulling pre-built image).
* **Accessibility:** Transformed a local project into a publicly shareable application via Streamlit Cloud.
