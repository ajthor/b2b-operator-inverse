## Results directory configuration

Training, evaluation, and plotting scripts write artifacts under a single base
directory that follows this precedence:

1. The `--base_dir` command-line option (when provided).
2. The `B2B_RESULTS_DIR` environment variable.
3. The fallback `./results` directory relative to the repository root.

Python entry points and shell launchers all use this same order, so you can pick
whichever mechanism best fits your workflow. For multi-user systems it is common
to export `B2B_RESULTS_DIR=/store/<user>/b2b_operator_inverse` (or similar) and
run the provided scripts without additional arguments. When experimenting
locally, simply omit the environment variable and the project will write to
`./results`.
