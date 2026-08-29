# Existing assets, licenses, and terms

This document records the public sources checked for the research assets used
by Delegent. It does not place third-party data, model weights, service output,
or index contents under this repository's Apache-2.0 license. Revisions below
identify audited public sources; where the historical experiment did not record
a revision, that limitation is stated explicitly.

| Asset | Audited identity | License or terms | Status and limitations |
|---|---|---|---|
| HotpotQA | [Official dataset](https://hotpotqa.github.io/) and current [code revision `3635853`](https://github.com/hotpotqa/hotpot/tree/3635853403a8735609ee997664e1528f4480762a) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | The exact source split and projected-holdout manifest used by the experiment are not released. No questions or qrels are redistributed here. |
| SQuAD 1.1 | [Official site](https://rajpurkar.github.io/SQuAD-explorer/), validation split, current [loader revision `7b6d24c`](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | The public artifact does not redistribute the frozen 992-question ID manifest. |
| NanoKnow | Current [code revision `76e4afc`](https://github.com/castorini/NanoKnow/tree/76e4afcbfe419de3155b24929065620a8233f47f) and [benchmark revision `57b1e60`](https://huggingface.co/datasets/castorini/NanoKnow_Benchmark/tree/57b1e60a4adb35bf92985066a105c09b3d7c458a) | Repository LICENSE and benchmark metadata say Apache-2.0 | Upstream `pyproject.toml` says MIT, so the metadata are inconsistent. Delegent follows the repository LICENSE and does not redistribute the projection. |
| FineWeb-Edu | [`karpathy/fineweb-edu-100b-shuffle`, revision `4c8f30d`](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/tree/4c8f30d6756da75362432a4d5569e1b229263b71) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) | ODC-By covers database rights; copyright in source webpages may remain with their owners. Corpus text is not redistributed here. |
| FineWeb-Edu Lucene index | [`lucene-inverted.fineweb-edu-100b-karpathy.20260416`, revision `a2334f2`](https://huggingface.co/datasets/castorini/NanoKnow-Fineweb-Edu-Index/tree/a2334f26854b7a5429c9803e7041400938eae09c) | Apache-2.0 for index packaging; underlying corpus remains ODC-By | The index card records BM25 parameters `k1=0.9`, `b=0.4`. |
| Pyserini, Anserini, Lucene | Public projects [Pyserini](https://github.com/castorini/pyserini), [Anserini](https://github.com/castorini/anserini), and [Lucene](https://github.com/apache/lucene) | Apache-2.0 | The exact historical service versions were not recorded and current heads are not claimed as experimental versions. |
| Gemma 4 E4B IT | [`google/gemma-4-E4B-it`, revision `ee0ef60`](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2) | [Gemma 4 Apache License 2.0](https://ai.google.dev/gemma/apache_2) | The released lineage binds the final continuation to this base revision. Adapter weights are not distributed here. |
| GPT-5.6 Sol | Hosted model documented on the [official model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol) | [OpenAI Services Agreement](https://openai.com/policies/services-agreement/) and [Usage Policies](https://openai.com/policies/usage-policies/) | This is a hosted service, not an open-licensed model. The release contains no service output. The exact access agreement governing teacher-output use must be confirmed before releasing trajectories or adapter weights. |
| BrowseComp-Plus | Current [code revision `0469490`](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402) and [dataset revision `144cff8`](https://huggingface.co/datasets/Tevatron/browsecomp-plus/tree/144cff8e35b5eaef7e526346aa60774a9deb941f) | MIT | The benchmark card asks that benchmark fields not be posted as plaintext. This release contains no questions, answers, qrels, or teacher trajectories. Source-page rights in corpus documents may remain. |

## Unresolved items

- Pin the historical HotpotQA source split and projection manifest.
- Pin the historical retrieval service's Pyserini, Anserini, Java, and Lucene
  versions.
- Release or bind the frozen SQuAD sample manifest without exposing evaluation
  content.
- Confirm the agreement governing GPT-5.6 Sol teacher-output use before any
  trajectory or adapter release.

These unresolved items are why the submission checklist does not claim full
existing-asset compliance or end-to-end reproducibility.
