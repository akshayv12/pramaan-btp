# Contributing

Set up the local environment using the README. Keep credentials, model weights, downloaded sources and personal reports out of commits.

Task 5's canonical source is `task5_standalone/task5/`. After changing it, run `python task5_standalone/sync_to_integrated.py`, then the offline test commands in the README. Include both copies of `support_policy.json` when changing decision rules. Do not change Tasks 1–4 algorithms or architecture as a side effect of a website or Task 5 change.

Add a focused regression check when fixing a substantive failure. Use constructed evidence or a short sanitised fixture with source attribution. Keep external API calls and large model downloads out of the default tests. Report which optional model checks you actually ran; mock responses do not establish factual accuracy.

For an issue, include the stage that failed, provider/model, Python/OS version and a sanitised error message. Do not paste API keys, private evidence or full search-history logs.
