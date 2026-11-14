# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/iamsettt1/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/iamsettt1/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One OpenAI/Anthropic-compatible gateway in front of ten simulated free-tier providers: it reads the trial brief (provider directory, free-model quality table, workload manifest), registers a catalog built from that brief, and answers every request with the best free model that meets its requirements — planning assignments against the providers' published budget ranges, passing each request's echo-relevant fields through wire translations untouched, failing over down a rating-ordered candidate list (with one in-range retry) whenever a provider refuses, and rendering streamed answers locally in the client's own wire with the answering provider's exact usage. Scores 99/100 on all three practice trials (served 100, wire 100, failover 100, catalog 100).

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9j3w1sg2&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
