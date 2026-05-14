"""mllmopd — local analysis package for the MLLM OPD research project.

This package runs on Mac for analysis / figure generation and on the devbox for
audit inference. Heavy GPU-only dependencies (torch, transformers, vllm) are
imported lazily inside functions so `import mllmopd` works on a CPU box.
"""

__version__ = "0.0.1"
