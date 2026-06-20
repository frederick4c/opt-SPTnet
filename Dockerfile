# opt-SPTnet: CPU-only image for reproducible installs, tests, and CLI use.
#
# This image installs the package and its CPU PyTorch build so the command-line
# tools and test suite run anywhere without a GPU or a MATLAB licence. For GPU
# training, start from an NVIDIA CUDA base image (for example
# nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04) and install the matching CUDA
# PyTorch wheel instead of the CPU index used below; the rest of the steps are
# unchanged.
FROM python:3.11-slim

# Keep Python output unbuffered and skip pip's version check noise.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/opt-sptnet

# Install the CPU PyTorch build first from the dedicated index so the generic
# dependency resolution below does not pull the much larger CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

# Copy project metadata and sources, then install the package with test extras.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
RUN pip install -e ".[test]"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 sptnet \
    && chown -R sptnet:sptnet /opt/opt-sptnet
USER sptnet

# Default to a harmless help command; override with any sptnet-* CLI.
CMD ["sptnet-train", "--help"]
