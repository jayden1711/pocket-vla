// Network state sampled DURING the offline speech test (CRITIQUES D4), plus the offline pass/fail rule.
// No UIKit/SwiftUI/Speech here, so it builds and is tested on macOS (app/Bench/NetTest/main.swift).

import Foundation
import Network

/// One observation of the network path.
struct NetSample: Codable, Equatable {
    let tMs: Double             // ms since the sampler's start() (monotonic clock)
    let date: String            // wall clock, ISO 8601 with fractional seconds
    let kind: String            // "start" | "tick" | "change" (path update) | "end"
    let pathKnown: Bool         // false: no path update received from NWPathMonitor yet (state unknown)
    let status: String          // "satisfied" | "unsatisfied" | "requiresConnection" | "unknown"
    let interfaceTypes: [String]  // per available interface: wifi | cellular | wired | loopback | other
    let interfaces: [String]      // "name(type)", e.g. "en0(wifi)"
    let isExpensive: Bool
    let isConstrained: Bool

    /// Conservative: anything but "unsatisfied" counts as usable: a path that is up ("satisfied"), one the
    /// system can bring up on demand ("requiresConnection"), or a status this code does not know.
    /// (Samples with no path yet are not "usable" here; OfflineCheck fails them separately.)
    var usable: Bool { pathKnown && status != "unsatisfied" }
}

extension NetSample {
    static func interfaceType(_ t: NWInterface.InterfaceType) -> String {
        switch t {
        case .wifi: return "wifi"
        case .cellular: return "cellular"
        case .wiredEthernet: return "wired"
        case .loopback: return "loopback"
        case .other: return "other"
        @unknown default: return "other"
        }
    }

    static func statusName(_ s: NWPath.Status) -> String {
        switch s {
        case .satisfied: return "satisfied"
        case .unsatisfied: return "unsatisfied"
        case .requiresConnection: return "requiresConnection"
        @unknown default: return "unknown"
        }
    }

    init(path: NWPath?, tMs: Double, date: String, kind: String) {
        self.tMs = tMs; self.date = date; self.kind = kind
        pathKnown = path != nil
        status = path.map { NetSample.statusName($0.status) } ?? "unknown"
        let ifs = path?.availableInterfaces ?? []
        interfaceTypes = ifs.map { NetSample.interfaceType($0.type) }
        interfaces = ifs.map { "\($0.name)(\(NetSample.interfaceType($0.type)))" }
        isExpensive = path?.isExpensive ?? false
        isConstrained = path?.isConstrained ?? false
    }
}

/// Offline verdict over a sample series. `offline` is true only if nothing in the series can be read as
/// "network might have been usable" and the series actually covers the recognition window.
struct OfflineCheck: Codable, Equatable {
    let offline: Bool
    let reasons: [String]          // empty when offline == true
    let sampleCount: Int
    let usableSampleCount: Int
    let maxGapMs: Double?          // largest gap between consecutive samples
    let allowedGapMs: Double

    /// Rules (all must hold):
    /// - at least one "start" sample and one "end" sample;
    /// - every sample has a known path (a path update was received) with status "unsatisfied";
    /// - no gap between consecutive samples is longer than `allowedGapMs`.
    static func evaluate(_ samples: [NetSample], allowedGapMs: Double = 1000) -> OfflineCheck {
        var reasons: [String] = []
        if samples.isEmpty { reasons.append("no network samples") }
        if !samples.contains(where: { $0.kind == "start" }) { reasons.append("no start sample") }
        if !samples.contains(where: { $0.kind == "end" }) { reasons.append("no end sample") }
        let unknown = samples.filter { !$0.pathKnown }
        if !unknown.isEmpty { reasons.append("\(unknown.count) sample(s) before the first path update (state unknown)") }
        let up = samples.filter(\.usable)
        if let first = up.first {
            reasons.append("\(up.count) sample(s) with a usable path (first at \(Int(first.tMs)) ms: "
                           + "\(first.status) via \(first.interfaces.joined(separator: ","))")
        }
        let ts = samples.map(\.tMs).sorted()
        let maxGap = zip(ts, ts.dropFirst()).map { $1 - $0 }.max()
        if let g = maxGap, g > allowedGapMs {
            reasons.append("sampling gap \(Int(g)) ms > \(Int(allowedGapMs)) ms")
        }
        return OfflineCheck(offline: reasons.isEmpty, reasons: reasons, sampleCount: samples.count,
                            usableSampleCount: samples.filter(\.usable).count, maxGapMs: maxGap,
                            allowedGapMs: allowedGapMs)
    }
}

/// Samples NWPathMonitor at start(), every `intervalMs` while running, on every path update, and at stop().
/// All state lives on one serial queue; the public methods are safe to call from any thread.
final class NetworkSampler: @unchecked Sendable {
    let intervalMs: Double
    private let queue = DispatchQueue(label: "pocketvla.netsampler")
    private let monitor = NWPathMonitor()
    private var latest: NWPath?       // nil until the first pathUpdateHandler call
    private var timer: DispatchSourceTimer?
    private var t0: UInt64 = 0
    private var samples: [NetSample] = []
    private var running = false
    private let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    init(intervalMs: Double = 250) {
        self.intervalMs = intervalMs
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            self.latest = path   // already on `queue`
            if self.running { self.record("change") }
        }
        monitor.start(queue: queue)
    }

    deinit { monitor.cancel(); timer?.cancel() }

    /// Waits (up to `timeout` s) until NWPathMonitor has delivered its first path, so the start sample is real.
    func waitForFirstPath(timeout: Double = 2) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if queue.sync(execute: { latest != nil }) { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return queue.sync { latest != nil }
    }

    func start() {
        queue.sync {
            samples = []
            t0 = DispatchTime.now().uptimeNanoseconds
            running = true
            record("start")
            let t = DispatchSource.makeTimerSource(queue: queue)
            let ns = Int(intervalMs * 1e6)
            t.schedule(deadline: .now() + .nanoseconds(ns), repeating: .nanoseconds(ns), leeway: .milliseconds(10))
            t.setEventHandler { [weak self] in self?.record("tick") }
            t.resume()
            timer = t
        }
    }

    /// Takes the end sample and returns the whole series. Returns [] if start() was never called.
    func stop() -> [NetSample] {
        queue.sync {
            guard running else { return samples }
            timer?.cancel(); timer = nil
            record("end")
            running = false
            return samples
        }
    }

    private func record(_ kind: String) {  // on `queue`
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        samples.append(NetSample(path: latest, tMs: ms, date: iso.string(from: Date()), kind: kind))
    }
}
