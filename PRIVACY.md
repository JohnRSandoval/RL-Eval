# Data and privacy

The public source release contains implementation code, generic configuration
examples, synthetic test fixtures, and documentation. It does not include recorded
matches, screenshots, model transcripts, API keys, local game files, or development
session history.

## Running the harness

API credentials are read from environment variables. No `.env` file is loaded
automatically. Provider requests use the configured endpoint; choosing a remote
provider sends that run's selected observations and context to that provider.
Pixels mode sends captured game images, state mode sends allowlisted measured
game state, and combined mode sends both. Model notes and tactical plans may also
be included in subsequent requests. State mode does not capture screenshots.
Model-free smoke and scripted-controller modes do not send model API requests.

Reports are written locally. Depending on the selected track, they include
screenshots, sampled positions, actions, model notes, tactical plans, timing, token
usage, configuration, and diagnostic logs. Configuration and diagnostics can
contain local paths or identifiers. The harness does not automatically upload
these reports to GitHub.

## Sharing changes or results

Keep API credentials outside source control. Use files named `*.local.toml` or
`configs/local/` for machine-specific configuration; both are ignored by Git.
Generated `runs/`, `calibration-runs/`, `runtime/`, environments, logs, and build
artifacts are ignored as well. Ignore rules do not protect files already tracked
by Git or files deliberately added with `git add --force`.

Review the exact files before uploading a report or committing a change. Game
screenshots and logs can expose account names or other information visible in
your session. The package's source distribution uses an explicit file allowlist,
but custom export tools and copied report folders need their own review.

The MIT license applies to this project's code. Rocket League and its assets,
RLBot, and provider services remain separate products under their own terms.
