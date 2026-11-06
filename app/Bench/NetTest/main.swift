// macOS test of the speech test's network sampler and offline pass/fail rule (CRITIQUES D4), no phone needed.
// Build + run from the repo root (or `make app-nettest`):
//   swiftc -o /tmp/nettest app/Bench/Sources/NetworkSampler.swift app/Bench/NetTest/main.swift && /tmp/nettest
// Part 1 checks OfflineCheck.evaluate on synthetic series. Part 2 runs the real NetworkSampler (NWPathMonitor)
// on this Mac for ~1.3 s and checks the series shape (start/ticks/end, spacing) and that the verdict agrees
// with the samples. Exit code 0 = all checks passed.

import Foundation

var failures = 0
func check(_ cond: Bool, _ name: String) {
    print((cond ? "PASS " : "FAIL ") + name)
    if !cond { failures += 1 }
}

func s(_ t: Double, _ kind: String, _ status: String = "unsatisfied", known: Bool = true,
       types: [String] = [], expensive: Bool = false, constrained: Bool = false) -> NetSample {
    NetSample(tMs: t, date: "2026-11-06T00:00:00.000Z", kind: kind, pathKnown: known, status: status,
              interfaceTypes: types, interfaces: types.enumerated().map { "if\($0.offset)(\($0.element))" },
              isExpensive: expensive, isConstrained: constrained)
}

/// start at 0, ticks every 250 ms up to `ms`, end at `ms` + 10.
func series(_ ms: Double, _ override: [Int: NetSample] = [:]) -> [NetSample] {
    var out = [s(0, "start")]
    var t = 250.0
    while t <= ms { out.append(s(t, "tick")); t += 250 }
    out.append(s(ms + 10, "end"))
    for (i, v) in override { out[i] = v }
    return out
}

// ---- Part 1: pass/fail rule on synthetic series ----
let clean = series(5000)
let ok = OfflineCheck.evaluate(clean)
check(ok.offline && ok.reasons.isEmpty && ok.usableSampleCount == 0, "all unsatisfied, 250 ms spacing -> offline")

check(!OfflineCheck.evaluate(series(5000, [7: s(1750, "tick", "satisfied", types: ["wifi"])])).offline,
      "one mid-run wifi sample satisfied -> not offline")
check(!OfflineCheck.evaluate(series(5000, [0: s(0, "start", "satisfied", types: ["cellular"], expensive: true)])).offline,
      "start sample on cellular -> not offline")
let endUp = series(5000)
check(!OfflineCheck.evaluate(Array(endUp.dropLast()) + [s(5010, "end", "satisfied", types: ["wired"])]).offline,
      "end sample on wired -> not offline")
check(!OfflineCheck.evaluate(series(5000, [3: s(750, "tick", "requiresConnection", types: ["other"])])).offline,
      "requiresConnection counts as usable -> not offline")
check(!OfflineCheck.evaluate(series(5000, [3: s(750, "tick", "unknown")])).offline,
      "unknown status counts as usable -> not offline")
let unknownPath = OfflineCheck.evaluate(series(5000, [0: s(0, "start", "unknown", known: false)]))
check(!unknownPath.offline && unknownPath.usableSampleCount == 0, "no path update yet at start -> not offline")
check(!OfflineCheck.evaluate([]).offline, "no samples -> not offline")
check(!OfflineCheck.evaluate(Array(clean.dropLast())).offline, "missing end sample -> not offline")
check(!OfflineCheck.evaluate(Array(clean.dropFirst())).offline, "missing start sample -> not offline")
let gappy = clean.filter { !($0.kind == "tick" && $0.tMs > 1000 && $0.tMs < 2500) }
let g = OfflineCheck.evaluate(gappy)
check(!g.offline && (g.maxGapMs ?? 0) > 1000, "sampling gap > 1 s -> not offline")
let change = series(5000) + [s(1234, "change", "satisfied", types: ["wifi"])]
check(!OfflineCheck.evaluate(change).offline, "path-change event to wifi between ticks -> not offline")
let loop = series(5000, [2: s(500, "tick", "unsatisfied", types: ["loopback"])])
check(OfflineCheck.evaluate(loop).offline, "unsatisfied with only loopback listed -> offline")

// JSON round trip (the saved result carries these types).
do {
    let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
    let data = try enc.encode(change)
    let back = try JSONDecoder().decode([NetSample].self, from: data)
    let v = try JSONDecoder().decode(OfflineCheck.self, from: try enc.encode(OfflineCheck.evaluate(change)))
    check(back == change && v == OfflineCheck.evaluate(change), "NetSample / OfflineCheck JSON round trip")
} catch {
    check(false, "JSON round trip threw \(error)")
}

// ---- Part 2: the real sampler on this Mac ----
let sampler = NetworkSampler(intervalMs: 250)
let sema = DispatchSemaphore(value: 0)
var live: [NetSample] = []
var gotPath = false
Task.detached {
    gotPath = await sampler.waitForFirstPath(timeout: 2)
    sampler.start()
    try? await Task.sleep(nanoseconds: 1_300_000_000)
    live = sampler.stop()
    sema.signal()
}
sema.wait()
let kinds = live.map(\.kind)
let ticks = kinds.filter { $0 == "tick" }.count
let v = OfflineCheck.evaluate(live)
check(gotPath, "live: NWPathMonitor delivered a path within 2 s")
check(kinds.first == "start" && kinds.last == "end", "live: series starts with start and ends with end")
check((4...6).contains(ticks), "live: 4-6 ticks in 1.3 s at 250 ms (got \(ticks))")
check((v.maxGapMs ?? .infinity) < 400, "live: max gap < 400 ms (got \(Int(v.maxGapMs ?? -1)) ms)")
check(live.allSatisfy { $0.pathKnown }, "live: every sample has a known path")
check(v.offline == live.allSatisfy { $0.status == "unsatisfied" }, "live: verdict agrees with the samples")
check(sampler.stop() == live, "live: second stop() returns the same series and adds nothing")
let first = live.first
print("live: \(live.count) samples, status \(first?.status ?? "?"), interfaces \(first?.interfaces ?? []), "
      + "expensive \(first?.isExpensive ?? false), constrained \(first?.isConstrained ?? false), offline=\(v.offline)"
      + (v.reasons.isEmpty ? "" : " (\(v.reasons.joined(separator: "; ")))"))

print(failures == 0 ? "ALL PASSED" : "\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
