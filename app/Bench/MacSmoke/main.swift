// macOS smoke test of the bench core (ModelBench.swift + System.swift) without Xcode or a phone.
// Mac numbers are a proxy only. Build + run from the repo root:
//   swiftc -O -o /tmp/bench_smoke app/Bench/Sources/ModelBench.swift app/Bench/Sources/System.swift \
//       app/Bench/MacSmoke/main.swift && /tmp/bench_smoke training/out/student_policy_fp16.mlpackage 50 CPU_AND_NE
import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: bench_smoke <model.mlpackage> [runs] [units]"); exit(2) }
let src = URL(fileURLWithPath: args[1])
let runs = args.count > 2 ? Int(args[2]) ?? 50 : 50
let unitName = args.count > 3 ? args[3] : "CPU_AND_NE"
let units = MLComputeUnits.choices.first { $0.label == unitName } ?? .cpuAndNeuralEngine

let sema = DispatchSemaphore(value: 0)
var compiled: URL?
Task {
    do { compiled = try await MLModel.compileModel(at: src) } catch { print("compile failed:", error) }
    sema.signal()
}
sema.wait()
guard let url = compiled else { exit(1) }
do {
    let r = try runBench(modelURL: url, config: BenchConfig(computeUnits: units, runs: runs),
                         progress: { _, _ in }, isCancelled: { false })
    let a = r.all!
    print(String(format: "%@ %@ load %.2fs first %.1fms median %.1fms p95 %.1fms hz %.1f peak %.0fMB thermal %@",
                 src.lastPathComponent, r.computeUnits, r.loadSeconds, r.firstPredictMs, a.medianMs, a.p95Ms, a.hz,
                 r.peakFootprintMB, r.thermalEnd))
} catch {
    print("bench failed:", error)
    exit(1)
}
