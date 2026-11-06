// Step 0 phone benchmark core: load a Core ML model, feed fixed inputs, time predictions.
// Platform-neutral (iOS + macOS) so it can be type-checked without Xcode.

import CoreML
import Foundation

struct BenchConfig {
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var runs: Int = 50              // used when sustainedSeconds == 0
    var sustainedSeconds: Double = 0 // 600 = the 10-min thermal run
    var chunkSeconds: Double = 1.6   // action chunk duration (student 16 steps @ 10 Hz; SmolVLA 50 @ 10 Hz = 5.0)
    var warmup: Int = 3
}

struct Sample: Codable {
    let t: Double          // seconds since the timed loop started
    let ms: Double         // prediction latency
    let thermal: String    // ProcessInfo.thermalState at that moment
    let footprintMB: Double
}

struct Stats: Codable {
    let n: Int
    let medianMs: Double
    let p95Ms: Double
    let maxMs: Double
    let hz: Double

    init?(_ ms: [Double]) {
        guard !ms.isEmpty else { return nil }
        let s = ms.sorted()
        n = s.count
        medianMs = Stats.percentile(s, 50)
        p95Ms = Stats.percentile(s, 95)
        maxMs = s.last!
        hz = 1000 / medianMs
    }

    /// Linear interpolation, same as numpy.percentile default.
    static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        if sorted.count == 1 { return sorted[0] }
        let r = p / 100 * Double(sorted.count - 1)
        let lo = Int(r.rounded(.down)), hi = Int(r.rounded(.up))
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (r - Double(lo))
    }
}

struct BenchResult: Codable {
    let model: String
    let computeUnits: String
    let device: String
    let os: String
    let date: String
    let loadSeconds: Double
    let firstPredictMs: Double
    let sustainedSeconds: Double
    let chunkSeconds: Double
    let all: Stats?
    /// Last 60 s of a sustained run ("p95 after 10 min").
    let lastMinute: Stats?
    let peakFootprintMB: Double
    let thermalStart: String
    let thermalEnd: String
    let thermalCounts: [String: Int]
    /// Pass bar (PLAN.md Step 0): p95 after 10 min <= 1 s AND <= 40% of the chunk duration.
    /// nil unless this was a >= 10-min sustained run.
    let passBar: Bool?
    let passBarRule: String
    let note: String
    /// Run settings + provenance (runs, warmup, input seed, Low Power Mode, app build, model I/O shapes),
    /// so a like-for-like comparison can be checked from the file alone.
    let settings: [String: String]
    let samples: [Sample]
}

enum BenchError: LocalizedError {
    case unsupportedInput(String)
    var errorDescription: String? {
        switch self { case .unsupportedInput(let s): return "Unsupported model input: \(s)" }
    }
}

extension MLComputeUnits {
    var label: String {
        switch self {
        case .cpuOnly: return "CPU_ONLY"
        case .cpuAndGPU: return "CPU_AND_GPU"
        case .cpuAndNeuralEngine: return "CPU_AND_NE"
        case .all: return "ALL"
        @unknown default: return "unknown"
        }
    }
    static let choices: [MLComputeUnits] = [.cpuAndNeuralEngine, .all, .cpuAndGPU, .cpuOnly]
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// Fixed, deterministic inputs for any multi-array model (same idea as eval/step0/bench_mac.py:
/// float inputs uniform [0,1) from a seeded generator, int inputs all ones). Latency does not depend on
/// values for these fixed-shape models; real tokens/images come with the closed-loop app.
func fixedInputs(for desc: MLModelDescription) throws -> MLFeatureProvider {
    var rng = SplitMix64(seed: 0)
    var dict: [String: MLFeatureValue] = [:]
    for (name, d) in desc.inputDescriptionsByName.sorted(by: { $0.key < $1.key }) {
        guard d.type == .multiArray, let c = d.multiArrayConstraint else { throw BenchError.unsupportedInput(name) }
        let arr = try MLMultiArray(shape: c.shape, dataType: c.dataType)
        let n = arr.count
        switch c.dataType {
        case .int32:
            let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: n)
            for i in 0..<n { p[i] = 1 }
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: n)
            for i in 0..<n { p[i] = Float32(rng.nextUnit()) }
        default:  // float16 / double: slow path, one-time
            for i in 0..<n { arr[i] = NSNumber(value: rng.nextUnit()) }
        }
        dict[name] = MLFeatureValue(multiArray: arr)
    }
    return try MLDictionaryFeatureProvider(dictionary: dict)
}

struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextUnit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

/// Runs one benchmark. Blocking: call from a background task. `progress` gets (fraction, latest ms).
func runBench(modelURL: URL, config: BenchConfig, progress: (Double, Double) -> Void,
              isCancelled: () -> Bool) throws -> BenchResult {
    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = config.computeUnits
    var peak = Memory.footprintMB()
    let thermalStart = ProcessInfo.processInfo.thermalState.label

    let t0 = Date()
    let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
    let loadSeconds = Date().timeIntervalSince(t0)
    let feeds = try fixedInputs(for: model.modelDescription)

    let tf = Clock.now()
    _ = try model.prediction(from: feeds)
    let firstMs = Clock.ms(since: tf)
    for _ in 0..<config.warmup { _ = try model.prediction(from: feeds) }
    peak = max(peak, Memory.footprintMB())

    var samples: [Sample] = []
    let start = Clock.now()
    let sustained = config.sustainedSeconds > 0
    while !isCancelled() {
        let elapsed = Clock.seconds(since: start)
        if sustained ? elapsed >= config.sustainedSeconds : samples.count >= config.runs { break }
        let t = Clock.now()
        _ = try autoreleasepool { try model.prediction(from: feeds) }
        let ms = Clock.ms(since: t)
        let fp = Memory.footprintMB()
        peak = max(peak, fp)
        samples.append(Sample(t: Clock.seconds(since: start), ms: ms,
                              thermal: ProcessInfo.processInfo.thermalState.label, footprintMB: fp))
        let frac = sustained ? elapsed / config.sustainedSeconds : Double(samples.count) / Double(config.runs)
        progress(min(frac, 1), ms)
    }

    let all = Stats(samples.map(\.ms))
    let endT = samples.last?.t ?? 0
    let lastMinute = sustained ? Stats(samples.filter { $0.t >= endT - 60 }.map(\.ms)) : nil
    var pass: Bool? = nil
    if sustained, config.sustainedSeconds >= 600, endT >= 599, let lm = lastMinute {
        pass = lm.p95Ms <= 1000 && lm.p95Ms <= 0.4 * config.chunkSeconds * 1000
    }
    var counts: [String: Int] = [:]
    for s in samples { counts[s.thermal, default: 0] += 1 }

    return BenchResult(
        model: modelURL.lastPathComponent, computeUnits: config.computeUnits.label,
        device: DeviceInfo.model, os: DeviceInfo.os, date: ISO8601DateFormatter().string(from: Date()),
        loadSeconds: loadSeconds, firstPredictMs: firstMs, sustainedSeconds: sustained ? endT : 0,
        chunkSeconds: config.chunkSeconds, all: all, lastMinute: lastMinute, peakFootprintMB: peak,
        thermalStart: thermalStart, thermalEnd: ProcessInfo.processInfo.thermalState.label,
        thermalCounts: counts, passBar: pass,
        passBarRule: "p95 over the last 60 s of a >=10-min run <= 1000 ms and <= 40% of chunkSeconds",
        note: isCancelled() ? "cancelled early" : "fixed synthetic inputs; app process footprint incl. UI",
        settings: runSettings(model: model, config: config),
        samples: samples)
}

func runSettings(model: MLModel, config: BenchConfig) -> [String: String] {
    func shapes(_ d: [String: MLFeatureDescription]) -> String {
        d.keys.sorted().map { k in
            let s = d[k]?.multiArrayConstraint?.shape.map { "\($0)" }.joined(separator: "x") ?? "?"
            return "\(k)=\(s)"
        }.joined(separator: ",")
    }
    let info = Bundle.main.infoDictionary ?? [:]
    return [
        "runs": "\(config.runs)", "warmup": "\(config.warmup)",
        "sustainedSecondsRequested": "\(config.sustainedSeconds)",
        "inputs": "float: SplitMix64(seed 0) uniform [0,1); int32: ones",
        "lowPowerMode": "\(ProcessInfo.processInfo.isLowPowerModeEnabled)",
        "appVersion": "\(info["CFBundleShortVersionString"] ?? "?") (\(info["CFBundleVersion"] ?? "?"))",
        "modelInputs": shapes(model.modelDescription.inputDescriptionsByName),
        "modelOutputs": shapes(model.modelDescription.outputDescriptionsByName),
    ]
}

enum Clock {
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func ms(since t: UInt64) -> Double { Double(now() - t) / 1e6 }
    static func seconds(since t: UInt64) -> Double { Double(now() - t) / 1e9 }
}

/// Writes the result as JSON into Documents/results/ (visible in the Files app / Finder file sharing).
@discardableResult
func save(_ r: BenchResult) throws -> URL {
    let dir = Locations.documents.appendingPathComponent("results", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let stamp = r.date.replacingOccurrences(of: ":", with: "-")
    let kind = r.sustainedSeconds > 0 ? "sustained" : "n\(r.all?.n ?? 0)"
    let url = dir.appendingPathComponent("\(r.model)_\(r.computeUnits)_\(kind)_\(stamp).json")
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    try enc.encode(r).write(to: url)
    return url
}
