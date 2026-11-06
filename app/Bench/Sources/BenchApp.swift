// pocket-vla Step 0 bench app: pick a model + compute units, run N predictions or a 10-min sustained
// run, and save JSON results to Documents/results/. Plus the offline speech check.

import CoreML
import SwiftUI
#if os(iOS)
import UIKit
#endif

@main
struct BenchApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class BenchModel: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published var selected: ModelEntry?
    @Published var units: MLComputeUnits = .cpuAndNeuralEngine
    @Published var runs = 50
    @Published var chunkSeconds = 1.6
    @Published var running = false
    @Published var progress = 0.0
    @Published var lastMs = 0.0
    @Published var status = ""
    @Published var result: BenchResult?

    private var job: Task<Void, Never>?
    private let cancelFlag = CancelFlag()

    func refresh() {
        models = discoverModels()
        if selected == nil || !models.contains(where: { $0 == selected }) { selected = models.first }
        if let s = selected { chunkSeconds = Self.defaultChunk(for: s.name) }
        status = models.isEmpty ? "No models. Bundle them via project.yml or copy .mlpackage into Documents." : ""
    }

    static func defaultChunk(for name: String) -> Double {
        name.lowercased().contains("smolvla") ? 5.0 : 1.6  // 50 vs 16 steps at 10 Hz
    }

    func start(sustainedSeconds: Double) {
        guard let entry = selected, !running else { return }
        running = true; progress = 0; result = nil; cancelFlag.value = false
        status = "compiling / loading \(entry.name)…"
        let cfg = BenchConfig(computeUnits: units, runs: runs, sustainedSeconds: sustainedSeconds,
                              chunkSeconds: chunkSeconds)
        setIdleTimer(disabled: true)
        let flag = cancelFlag
        job = Task.detached(priority: .userInitiated) { [self] in
            do {
                let url = try await compiledURL(for: entry)
                await MainActor.run { self.status = "running…" }
                let r = try runBench(modelURL: url, config: cfg, progress: { frac, ms in
                    Task { @MainActor in self.progress = frac; self.lastMs = ms }
                }, isCancelled: { flag.value })
                let saved = try save(r)
                await MainActor.run { self.result = r; self.status = "saved \(saved.lastPathComponent)" }
            } catch {
                await MainActor.run { self.status = "error: \(error.localizedDescription)" }
            }
            await MainActor.run {
                self.running = false
                self.setIdleTimer(disabled: false)
            }
        }
    }

    func cancel() { cancelFlag.value = true }

    private func setIdleTimer(disabled: Bool) {
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = disabled  // screen must stay on for the 10-min run
        #endif
    }
}

/// Thread-safe flag read from the benchmark loop.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    var value: Bool {
        get { lock.lock(); defer { lock.unlock() }; return v }
        set { lock.lock(); v = newValue; lock.unlock() }
    }
}

struct ContentView: View {
    @StateObject private var bench = BenchModel()
    @StateObject private var speech = SpeechTest()

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Model", selection: $bench.selected) {
                        ForEach(bench.models) { m in Text("\(m.name) (\(m.source))").tag(Optional(m)) }
                    }
                    Picker("Compute units", selection: $bench.units) {
                        ForEach(MLComputeUnits.choices, id: \.rawValue) { u in Text(u.label).tag(u) }
                    }
                    Stepper("Runs: \(bench.runs)", value: $bench.runs, in: 10...500, step: 10)
                    Stepper(String(format: "Chunk: %.1f s", bench.chunkSeconds), value: $bench.chunkSeconds,
                            in: 0.4...10, step: 0.2)
                    Button("Reload model list") { bench.refresh() }
                }
                Section("Run") {
                    Button("Run \(bench.runs) predictions") { bench.start(sustainedSeconds: 0) }
                        .disabled(bench.running || bench.selected == nil)
                    Button("Sustained 10 min (thermal)") { bench.start(sustainedSeconds: 600) }
                        .disabled(bench.running || bench.selected == nil)
                    if bench.running {
                        ProgressView(value: bench.progress)
                        Text(String(format: "last %.1f ms · %@", bench.lastMs,
                                    ProcessInfo.processInfo.thermalState.label))
                        Button("Stop", role: .destructive) { bench.cancel() }
                    }
                    if !bench.status.isEmpty { Text(bench.status).font(.footnote) }
                }
                if let r = bench.result { ResultSection(r: r) }
                Section("Offline speech (airplane mode, Bluetooth on)") {
                    Button(speech.running ? "Listening…" : "Record 5 s, on-device only") {
                        Task { await speech.run() }
                    }.disabled(speech.running)
                    Text(speech.status).font(.footnote)
                    if !speech.transcript.isEmpty { Text("“\(speech.transcript)”") }
                }
                Section {
                    Text("\(DeviceInfo.model) · \(DeviceInfo.os)").font(.footnote)
                }
            }
            .navigationTitle("pocket-vla bench")
            .onAppear { bench.refresh() }
            .onChange(of: bench.selected) { _, s in
                if let s { bench.chunkSeconds = BenchModel.defaultChunk(for: s.name) }
            }
        }
    }
}

struct ResultSection: View {
    let r: BenchResult
    var body: some View {
        Section("Result: \(r.model) · \(r.computeUnits)") {
            row("load", String(format: "%.2f s", r.loadSeconds))
            row("first predict", String(format: "%.1f ms", r.firstPredictMs))
            if let a = r.all {
                row("median / p95", String(format: "%.1f / %.1f ms", a.medianMs, a.p95Ms))
                row("Hz (median)", String(format: "%.2f", a.hz))
                row("n", "\(a.n)")
            }
            if let lm = r.lastMinute { row("last-minute p95", String(format: "%.1f ms", lm.p95Ms)) }
            row("peak footprint", String(format: "%.0f MB", r.peakFootprintMB))
            row("thermal", "\(r.thermalStart) → \(r.thermalEnd)")
            if let p = r.passBar { row("pass bar", p ? "PASS" : "FAIL") }
        }
    }
    func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).monospacedDigit().foregroundStyle(.secondary) }
    }
}
