// Memory footprint, device identity, model discovery.

import CoreML
import Foundation

enum Memory {
    /// phys_footprint: what iOS's memory limit (jetsam) counts, same number as Xcode's memory gauge.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }
}

enum DeviceInfo {
    /// Hardware identifier, e.g. "iPhone17,1" (map to a marketing name when filling PLAN.md).
    static var model: String {
        var sys = utsname()
        uname(&sys)
        return withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
    static var os: String { ProcessInfo.processInfo.operatingSystemVersionString }
}

enum Locations {
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var compiledCache: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("compiled", isDirectory: true)
    }
}

struct ModelEntry: Identifiable, Hashable {
    let url: URL          // a compiled .mlmodelc (bundled) or a source .mlpackage/.mlmodel (Documents)
    let source: String    // "bundle" or "documents"
    var id: URL { url }
    var name: String { url.deletingPathExtension().lastPathComponent }
}

/// Models compiled into the app bundle by Xcode (from training/out/, see project.yml) plus any
/// .mlpackage / .mlmodel copied into Documents via Finder file sharing (compiled on device on first use).
func discoverModels() -> [ModelEntry] {
    var out: [ModelEntry] = []
    let fm = FileManager.default
    if let res = Bundle.main.resourceURL,
       let items = try? fm.contentsOfDirectory(at: res, includingPropertiesForKeys: nil) {
        // .mlmodelc when Xcode compiled it; a raw .mlpackage (if it ended up in Copy Resources) is compiled on device
        out += items.filter { ["mlmodelc", "mlpackage"].contains($0.pathExtension) }
            .map { ModelEntry(url: $0, source: "bundle") }
    }
    if let items = try? fm.contentsOfDirectory(at: Locations.documents, includingPropertiesForKeys: nil) {
        out += items.filter { ["mlpackage", "mlmodel"].contains($0.pathExtension) }
            .map { ModelEntry(url: $0, source: "documents") }
    }
    return out.sorted { $0.name < $1.name }
}

/// Returns a loadable .mlmodelc URL, compiling (and caching) Documents models on device.
func compiledURL(for entry: ModelEntry) async throws -> URL {
    if entry.url.pathExtension == "mlmodelc" { return entry.url }
    let fm = FileManager.default
    try fm.createDirectory(at: Locations.compiledCache, withIntermediateDirectories: true)
    let dest = Locations.compiledCache.appendingPathComponent(entry.name + ".mlmodelc")
    let srcDate = (try? fm.attributesOfItem(atPath: entry.url.path)[.modificationDate] as? Date) ?? .distantPast
    let dstDate = (try? fm.attributesOfItem(atPath: dest.path)[.modificationDate] as? Date) ?? .distantPast
    if fm.fileExists(atPath: dest.path), dstDate >= srcDate { return dest }
    let tmp = try await MLModel.compileModel(at: entry.url)
    if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
    try fm.moveItem(at: tmp, to: dest)
    return dest
}
