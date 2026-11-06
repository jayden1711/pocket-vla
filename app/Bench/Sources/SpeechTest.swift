// Offline speech check for the demo: on-device SFSpeechRecognizer with network off.
// Run it in airplane mode. The network path is sampled at recognition start, every 250 ms while it runs,
// on every path change, and at the end (NetworkSampler.swift); the whole series is saved with the result and
// the result fails the offline check if any sample shows a usable path (OfflineCheck.evaluate).

import AVFoundation
import Foundation
import Network
import Speech

struct SpeechResult: Codable {
    let date: String
    let locale: String
    let supportsOnDevice: Bool
    let offlineCheck: OfflineCheck       // offlineCheck.offline must be true for a valid airplane-mode test
    let networkAvailable: Bool           // true if ANY sample showed a usable path
    let networkInterfaces: [String]      // every interface seen in any sample
    let networkSampleIntervalMs: Double
    let networkSamples: [NetSample]      // full series: start, ticks, path changes, end
    let transcript: String
    let firstPartialMs: Double?  // mic start -> first partial transcript
    let finalMs: Double?         // stop -> final transcript
    let error: String?
}

@MainActor
final class SpeechTest: ObservableObject {
    @Published var status = "idle"
    @Published var transcript = ""
    @Published var running = false

    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recognizer: SFSpeechRecognizer?
    private var start: UInt64 = 0
    private var stopAt: UInt64 = 0
    private var firstPartialMs: Double?
    private var finalMs: Double?
    private var finished = false
    private let sampler = NetworkSampler(intervalMs: 250)  // monitor starts now, so a path is known by run()

    func requestAuth() async -> Bool {
        let speech = await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) }
        }
        let mic = await AVCaptureDevice.requestAccess(for: .audio)
        return speech && mic
    }

    /// Records for `seconds`, then stops and saves a SpeechResult JSON in Documents/results/.
    func run(seconds: Double = 5, locale: Locale = Locale(identifier: "en-US")) async {
        guard !running else { return }
        running = true
        defer { running = false }
        transcript = ""; firstPartialMs = nil; finalMs = nil; finished = false
        guard await requestAuth() else { status = "speech or mic permission denied"; return }
        guard let rec = SFSpeechRecognizer(locale: locale) else { status = "no recognizer for \(locale.identifier)"; return }
        recognizer = rec
        guard rec.supportsOnDeviceRecognition else {
            status = "on-device recognition NOT supported for \(locale.identifier)"
            saveResult(error: "supportsOnDeviceRecognition == false", samples: [])
            return
        }
        // If no path arrives in time the samples are marked pathKnown=false and the offline check fails.
        _ = await sampler.waitForFirstPath(timeout: 2)
        sampler.start()  // "start" sample: recognition start
        do {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            #endif
            let req = SFSpeechAudioBufferRecognitionRequest()
            req.requiresOnDeviceRecognition = true  // never send audio to Apple's servers
            req.shouldReportPartialResults = true
            request = req
            let input = engine.inputNode
            input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { buf, _ in
                req.append(buf)
            }
            engine.prepare()
            try engine.start()
            start = Clock.now()
            status = "listening \(Int(seconds)) s: say \"go to the red chair\""
            task = rec.recognitionTask(with: req) { [weak self] result, error in
                let owner = self
                Task { @MainActor in owner?.handle(result: result, error: error) }
            }
            try await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
            stopAt = Clock.now()
            engine.stop()
            input.removeTap(onBus: 0)
            req.endAudio()
            status = "waiting for final transcript"
            for _ in 0..<50 where !finished { try await Task.sleep(nanoseconds: 100_000_000) }
            let samples = sampler.stop()  // "end" sample: final transcript (or timeout)
            saveResult(error: finished ? nil : "no final result within 5 s", samples: samples)
        } catch {
            let samples = sampler.stop()
            engine.stop()
            status = "error: \(error.localizedDescription)"
            saveResult(error: error.localizedDescription, samples: samples)
        }
    }

    private func handle(result: SFSpeechRecognitionResult?, error: Error?) {
        if let r = result {
            transcript = r.bestTranscription.formattedString
            if firstPartialMs == nil { firstPartialMs = Clock.ms(since: start) }
            if r.isFinal { finished = true; if stopAt > 0 { finalMs = Clock.ms(since: stopAt) } }
        }
        if error != nil, !finished { finished = true }
    }

    private func saveResult(error: String?, samples: [NetSample]) {
        let check = OfflineCheck.evaluate(samples)
        var ifaces: [String] = []
        for s in samples { for i in s.interfaces where !ifaces.contains(i) { ifaces.append(i) } }
        let r = SpeechResult(
            date: ISO8601DateFormatter().string(from: Date()), locale: recognizer?.locale.identifier ?? "?",
            supportsOnDevice: recognizer?.supportsOnDeviceRecognition ?? false,
            offlineCheck: check, networkAvailable: samples.contains(where: \.usable), networkInterfaces: ifaces,
            networkSampleIntervalMs: sampler.intervalMs, networkSamples: samples,
            transcript: transcript, firstPartialMs: firstPartialMs, finalMs: finalMs, error: error)
        do {
            let dir = Locations.documents.appendingPathComponent("results", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent("speech_\(r.date.replacingOccurrences(of: ":", with: "-")).json")
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(r).write(to: url)
            let net = check.offline
                ? "offline check PASSED (\(check.sampleCount) samples)"
                : "offline check FAILED, not a valid offline test: \(check.reasons.joined(separator: "; "))"
            status = "saved (\(net)): \(error ?? "ok")"
        } catch {
            status = "could not save: \(error.localizedDescription)"
        }
    }
}
