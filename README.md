# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/iamsettt1/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/iamsettt1/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One gateway speaking both the OpenAI Chat Completions and the Anthropic Messages APIs (JSON and SSE) in front of ten simulated free-tier providers. It reads the trial brief — the provider directory, the free-model quality table and the workload manifest — registers a catalog built from that brief alone, and then routes every request by a deterministic auction: each candidate model is scored as its rating minus the probability that its provider's next call refuses (under the instance generator's own budget, failure-schedule and outage distributions) times a fixed call value over the request's weight. Providers whose refusals say the budget is spent or that they are out are retired on the strength of the error message itself; a failure-schedule refusal earns one immediate retry on the same candidate where the provider's budget range is wide. The density penalty fades up as the running weighted shortfall nears the quality-99 band edge, and a relaxed second phase guarantees no request is left unanswered. Each answer is rendered locally in the client's wire from one non-streamed provider call, with the provider's echo text whole and its exact usage counts.

Scores 99/100 on all three practice trials (quality 99; served, wire, failover and catalog 100) using 71 provider calls in total, replays byte-for-byte, and holds 99 on 40 additional generator draws at 23.7 calls per trial on average.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9j3w1sg2&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
