# Contributing

Set up the local environment using the README. Keep credentials, model weights, downloaded sources and personal reports out of commits.

Task 5's single source is `task5/`, imported by the website and pipeline. Run the offline test commands in the README after changing it. Keep `support_policy.json` in the package when changing decision rules. There is no mirrored implementation or sync step. Each task owns its code and README; orchestration belongs in `pipeline/`. Do not change Tasks 1–4 algorithms as a side effect of a website or Task 5 change.

Add a focused regression check when fixing a substantive failure. Use constructed evidence or a short sanitised fixture with source attribution. Keep external API calls and large model downloads out of the default tests. Report which optional model checks you actually ran; mock responses do not establish factual accuracy.

For an issue, include the stage that failed, provider/model, Python/OS version and a sanitised error message. Do not paste API keys, private evidence or full search-history logs.
