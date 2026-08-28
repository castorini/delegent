# Routed-family configuration

`hotpotqa-c10.json` is the secret-free template for the fixed ten-slot HotpotQA family. Paths are resolved relative to the configuration file. The schema is:

```json
{
  "schema_version": "delegent-routed-family-config-v1",
  "source_records": "path/to/canonical-records",
  "output_dir": "path/to/routed-output",
  "cap": 10,
  "frontier_actor": "frontier-label-in-source-records",
  "delegate_actor": "delegate-label-for-new-records",
  "frontier_prefixes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "prompt": {
    "context_hits_per_search": 1,
    "passage_tokens": 160
  },
  "generation": {
    "max_attempts": 2,
    "deterministic_repair": true
  }
}
```

A live local copy additionally requires:

```json
{
  "retrieval": {
    "base_url": "https://your-retriever.example",
    "index": "your-index",
    "hits": 5,
    "timeout_seconds": 60,
    "token_env": "YOUR_RETRIEVAL_TOKEN_ENV"
  },
  "delegate": {
    "base_url": "https://your-openai-compatible-endpoint.example/v1",
    "model": "your-served-model-identity",
    "max_tokens": 256,
    "timeout_seconds": 120,
    "api_key_env": "YOUR_DELEGATE_API_KEY_ENV"
  }
}
```

The committed configuration deliberately omits deployment URLs and environment-variable names. Put them only in `configs/*.local.json`, which is ignored. Never store credential values in a configuration file.
