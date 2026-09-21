# Copy this directory to a host path (this kit: /home/spark/glm53-deploy) and
# edit the values below. Helper scripts source this file from the same directory.
#
# Defaults match start.sh: glm53-exl3-head / glm53-exl3-worker, API :8888,
# served id GLM-5.3-Flash-EXL3. A kit that preserves an older local name can
# set CONTAINER_HEAD / CONTAINER_WORKER in the repo .env to the same name on
# each node — Docker namespaces are per host.

GLM53_USER="${GLM53_USER:-spark}"
GLM53_DEPLOY_DIR="${GLM53_DEPLOY_DIR:-/home/spark/glm53-deploy}"
GLM53_REPO="${GLM53_REPO:-/home/spark/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
GLM53_CONTAINER_HEAD="${GLM53_CONTAINER_HEAD:-glm53-exl3-head}"
GLM53_CONTAINER_WORKER="${GLM53_CONTAINER_WORKER:-glm53-exl3-worker}"
GLM53_WORKER_SSH="${GLM53_WORKER_SSH:-spark@spark-worker}"
GLM53_WORKER_UNIT="${GLM53_WORKER_UNIT:-glm53-worker.service}"
GLM53_KNOWN_HOSTS="${GLM53_KNOWN_HOSTS:-/home/${GLM53_USER}/.ssh/known_hosts}"
GLM53_PORT="${GLM53_PORT:-8888}"
GLM53_SERVED_MODEL_NAME="${GLM53_SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
GLM53_HEALTH_TIMEOUT_SEC="${GLM53_HEALTH_TIMEOUT_SEC:-1800}"
GLM53_WORKER_WAIT_SEC="${GLM53_WORKER_WAIT_SEC:-1800}"
GLM53_COORD_TIMEOUT_SEC="${GLM53_COORD_TIMEOUT_SEC:-600}"
