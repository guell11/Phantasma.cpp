# Gremlin orchestration heartbeat

Chat On Steroids agent families are isolated by prime/conversation. A status
query from a different or unattributed conversation can therefore report no
agents even while workers from another prime are still active. Do not treat an
empty local agent list as proof that the project is idle.

The workspace is the cross-prime coordination surface. Every worker started by
the hourly coordinator must own exactly one lane and maintain one heartbeat
file in this directory:

- `perf-runtime.state`
- `treedraft-bonsai.state`
- `evidence-lab.state`

Each state file is plain UTF-8 text with these fields:

```text
lane=<lane name>
state=active|finished|blocked
task=<short concrete task>
updated=<ISO-8601 local timestamp>
evidence=<files/tests/build currently being touched>
```

Workers update their lane file when they start, after each material milestone,
before a long build/test, and immediately before they finish. A worker must not
write another lane's heartbeat.

## Hourly watchdog rule

The coordinator first checks:

1. agent status for the family it actually owns;
2. these heartbeat files;
3. recent source/test/document changes;
4. active `cmake`, `MSBuild`, `ninja`, `cl` and `nvcc` processes.

`no agent family belongs to this conversation`, `Unattributed`, or an empty
local agent list is **not** a death signal by itself.

A lane may be revived only when its heartbeat is `finished`/`blocked`, is
missing, or is stale for more than 90 minutes **and** there is no matching
build/test process or recent materialized activity suggesting that lane is
still working. Never exceed three concurrent specialist lanes.

When uncertain, prefer not to spawn a duplicate. The next hourly pass can
re-evaluate with more evidence.

