import Foundation
import XCTest

@testable import FluidAudio

final class ForkPatchTests: XCTestCase {
    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ForkPatchTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TreeStubURLProtocol.self]
        return config
    }

    private func makeUnmarkedBundle(_ path: String) throws {
        let bundle = workDir.appendingPathComponent(path)
        try CompiledBundleFixture.writeComplete(at: bundle)
        try FileManager.default.removeItem(at: bundle.appendingPathComponent(ModelCache.completeMarkerName))
    }

    private func stubBundlePack(bodySize: Int) {
        TreeStubURLProtocol.trees = [
            "pack": [
                ["path": "pack/encoder.mlmodelc", "type": "directory"],
                ["path": "pack/q8", "type": "directory"],
                ["path": "pack/vocab.json", "type": "file", "size": bodySize],
            ],
            "pack/encoder.mlmodelc": [
                ["path": "pack/encoder.mlmodelc/coremldata.bin", "type": "file", "size": bodySize],
                ["path": "pack/encoder.mlmodelc/model.mil", "type": "file", "size": bodySize],
                ["path": "pack/encoder.mlmodelc/weights", "type": "directory"],
            ],
            "pack/encoder.mlmodelc/weights": [
                ["path": "pack/encoder.mlmodelc/weights/weight.bin", "type": "file", "size": bodySize]
            ],
            "pack/q8": [["path": "pack/q8/decoder.mlmodelc", "type": "directory"]],
            "pack/q8/decoder.mlmodelc": [
                ["path": "pack/q8/decoder.mlmodelc/coremldata.bin", "type": "file", "size": bodySize],
                ["path": "pack/q8/decoder.mlmodelc/model.mil", "type": "file", "size": bodySize],
            ],
        ]
        TreeStubURLProtocol.fileBody = Data((String(repeating: " ", count: bodySize - 1) + "}").utf8)
    }

    func testBundleWithoutCompleteMarkerIsIncomplete() throws {
        try makeUnmarkedBundle("encoder.mlmodelc")

        XCTAssertEqual(
            ModelCache.incompleteFiles(at: workDir, requiredFiles: ["encoder.mlmodelc"]),
            ["encoder.mlmodelc"])
    }

    func testMarkBundlesCompleteMarksEachBundleOnDiskOnce() throws {
        try makeUnmarkedBundle("encoder.mlmodelc")
        try makeUnmarkedBundle("q8/decoder.mlmodelc")
        try ModelCache.markBundlesComplete(
            at: workDir,
            localPaths: [
                "encoder.mlmodelc/coremldata.bin",
                "encoder.mlmodelc/weights/weight.bin",
                "q8/decoder.mlmodelc/model.mil",
                "missing.mlmodelc/coremldata.bin",
                "vocab.json",
            ])

        let fm = FileManager.default
        XCTAssertTrue(
            fm.fileExists(
                atPath: workDir.appendingPathComponent("encoder.mlmodelc/\(ModelCache.completeMarkerName)").path))
        XCTAssertTrue(
            fm.fileExists(
                atPath: workDir.appendingPathComponent("q8/decoder.mlmodelc/\(ModelCache.completeMarkerName)").path))
        XCTAssertFalse(fm.fileExists(atPath: workDir.appendingPathComponent("missing.mlmodelc").path))
        XCTAssertFalse(
            fm.fileExists(atPath: workDir.appendingPathComponent(ModelCache.completeMarkerName).path))
    }

    func testFinishedDownloadMarksEveryBundleIncludingOnesAlreadyOnDisk() async throws {
        stubBundlePack(bodySize: 16)
        let decoder = workDir.appendingPathComponent("pack/q8/decoder.mlmodelc")
        try FileManager.default.createDirectory(at: decoder, withIntermediateDirectories: true)
        try Data(String(repeating: "x", count: 16).utf8).write(
            to: decoder.appendingPathComponent("coremldata.bin"))

        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            configuration: stubConfiguration)

        let pack = workDir.appendingPathComponent("pack")
        XCTAssertTrue(
            ModelCache.isCacheComplete(
                at: pack, requiredFiles: ["encoder.mlmodelc", "q8/decoder.mlmodelc", "vocab.json"]))
    }

    func testFailedDownloadMarksNoBundle() async throws {
        stubBundlePack(bodySize: 16)
        let encoder = workDir.appendingPathComponent("pack/encoder.mlmodelc")
        try FileManager.default.createDirectory(at: encoder, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: encoder.appendingPathComponent("coremldata.bin"))
        try Data("x".utf8).write(to: encoder.appendingPathComponent("weights"))

        do {
            try await ModelHub.download(
                .vad, subdirectory: "pack", to: workDir,
                configuration: stubConfiguration)
            XCTFail("download should throw on the blocked weights path")
        } catch {}

        for bundle in ["pack/encoder.mlmodelc", "pack/q8/decoder.mlmodelc"] {
            let marker = workDir.appendingPathComponent(bundle).appendingPathComponent(ModelCache.completeMarkerName)
            XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path), bundle)
        }
        XCTAssertFalse(
            ModelCache.isCacheComplete(
                at: workDir.appendingPathComponent("pack"), requiredFiles: ["encoder.mlmodelc"]))
    }

    func testDownloadClearsALeftoverPartialOfAFileAlreadyOnDisk() async throws {
        stubBundlePack(bodySize: 16)
        let weights = workDir.appendingPathComponent("pack/encoder.mlmodelc/weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        let weight = weights.appendingPathComponent("weight.bin")
        try Data(String(repeating: "x", count: 16).utf8).write(to: weight)
        let partial = weight.appendingPathExtension("partial")
        try Data("xxxx".utf8).write(to: partial)
        try Data("\"etag\"".utf8).write(to: FileDownloader.resumeValidatorURL(for: partial))

        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            configuration: stubConfiguration)

        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: FileDownloader.resumeValidatorURL(for: partial).path))
        XCTAssertTrue(
            ModelCache.isCacheComplete(
                at: workDir.appendingPathComponent("pack"), requiredFiles: ["encoder.mlmodelc"]))
    }
}

final class ModelLayoutTests: XCTestCase {
    private final class ListingServer {
        private let repo: String
        private let pages: [String: [[String: Any]]]
        private(set) var requested: [String] = []

        init(repo: String, pages: [String: [[String: Any]]]) {
            self.repo = repo
            self.pages = pages
        }

        var fetch: HFTreeLister.Fetch {
            { url in
                let apiPath = self.pages.keys.first { (try? ModelRegistry.apiModels(self.repo, $0)) == url }
                self.requested.append(apiPath ?? url.absoluteString)
                guard let apiPath, let items = self.pages[apiPath] else { throw DownloadError.invalidResponse }
                return (
                    try JSONSerialization.data(withJSONObject: items),
                    HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: [:])!
                )
            }
        }
    }

    private let otherRepository = "someone/turn-detector-coreml"
    private let pinnedRevision = String(repeating: "a", count: 40)

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelLayoutTests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    private func write(_ path: String, bytes: Int = 4) throws {
        let url = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 1, count: bytes).write(to: url)
    }

    private func writeFinishedBundle(_ path: String) throws {
        try CompiledBundleFixture.writeComplete(at: root.appendingPathComponent(path))
    }

    private func relative(_ url: URL) -> String {
        XCTAssertTrue(url.path.hasPrefix(root.path + "/"), "\(url.path) is not spelled under the root it was found in")
        return String(url.path.dropFirst(root.path.count + 1))
    }

    private func exists(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent(path).path)
    }

    private func folder(_ path: String, _ entries: Set<String>) -> ModelFolder {
        ModelFolder(url: root.appendingPathComponent(path), requiredEntries: entries, repo: .vad)
    }

    private func turnDetectorFolder() -> ModelFolder {
        ModelFolder(
            url: root.appendingPathComponent("turn-detector"),
            requiredEntries: [
                "detector.mlmodelc/coremldata.bin", "detector.mlmodelc/model.mil",
                "detector.mlmodelc/weights/weight.bin",
            ],
            remoteRepoPath: otherRepository, revision: pinnedRevision, remoteDirectory: "")
    }

    private func pocketTtsPack() throws -> ModelFolder {
        let layout = try XCTUnwrap(ModelLayout.pocketTts().first)
        return ModelFolder(
            url: root.appendingPathComponent("pack"), requiredEntries: layout.requiredEntries, repo: .pocketTts,
            remoteDirectory: layout.remoteDirectory, shouldSkip: layout.shouldSkip)
    }

    func testRemovesBundlesAndModelFoldersNoCurrentModelUses() throws {
        try writeFinishedBundle("silero-vad/silero-vad-unified-256ms-v6.2.1.mlmodelc")
        try writeFinishedBundle("silero-vad/silero-vad-unified-256ms-v6.0.0.mlmodelc")
        try writeFinishedBundle("pocket-tts/mimi_encoderv2.mlmodelc")
        try writeFinishedBundle("pocket-tts/v2.1/english_2026-10/flowlm_step.mlmodelc")
        try write("pocket-tts/v2.1/english_2026-10/constants_bin/alba.safetensors")
        try writeFinishedBundle("pocket-tts/v2.1/english_2026-04/flowlm_step.mlmodelc")
        try writeFinishedBundle("pocket-tts/v2/english/flowlm_step.mlmodelc")
        try write("pocket-tts/encoder_recover_pinv.bin")
        try writeFinishedBundle("retired-tts/ane/TextEncoder.mlmodelc")
        try write("retired-tts/tokens.txt")
        try writeFinishedBundle("diarizer/model.mlmodelc")
        try write("notes/readme.json")

        let removed = ModelLayout.removeUnusedModels(
            under: [root],
            keeping: [
                folder("silero-vad", ["silero-vad-unified-256ms-v6.2.1.mlmodelc"]),
                folder("pocket-tts/v2.1/english_2026-10", ["flowlm_step.mlmodelc", "constants_bin/alba.safetensors"]),
            ]
        )

        XCTAssertEqual(
            Set(removed.map(relative)),
            [
                "silero-vad/silero-vad-unified-256ms-v6.0.0.mlmodelc",
                "pocket-tts/mimi_encoderv2.mlmodelc",
                "pocket-tts/v2.1/english_2026-04",
                "pocket-tts/v2",
                "retired-tts",
                "diarizer",
            ]
        )
        for kept in [
            "silero-vad/silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin",
            "pocket-tts/v2.1/english_2026-10/flowlm_step.mlmodelc/coremldata.bin",
            "pocket-tts/v2.1/english_2026-10/constants_bin/alba.safetensors",
            "pocket-tts/encoder_recover_pinv.bin",
            "notes/readme.json",
        ] {
            XCTAssertTrue(exists(kept), "\(kept) should stay")
        }
    }

    func testKeepsEverythingWhenAllModelsAreCurrent() throws {
        try writeFinishedBundle("silero-vad/silero-vad-unified-256ms-v6.2.1.mlmodelc")
        try writeFinishedBundle("parakeet-eou-streaming/parakeet-eou-streaming/160ms/decoder.mlmodelc")
        try write("parakeet-eou-streaming/parakeet-eou-streaming/160ms/vocab.json")
        for file in turnDetectorFolder().requiredEntries {
            try write("turn-detector/\(file)")
        }

        let removed = ModelLayout.removeUnusedModels(
            under: [root],
            keeping: [
                folder("silero-vad", ["silero-vad-unified-256ms-v6.2.1.mlmodelc"]),
                folder("parakeet-eou-streaming/parakeet-eou-streaming/160ms", ["decoder.mlmodelc", "vocab.json"]),
                turnDetectorFolder(),
            ]
        )

        XCTAssertEqual(removed, [])
        XCTAssertTrue(exists("parakeet-eou-streaming/parakeet-eou-streaming/160ms/decoder.mlmodelc/coremldata.bin"))
        XCTAssertTrue(exists("turn-detector/detector.mlmodelc/weights/weight.bin"))
    }

    func testSearchesRootsWhosePathsDifferOnlyByCase() throws {
        try writeFinishedBundle("fluidaudio/Models/pocket-tts/flowlm_step.mlmodelc")
        try writeFinishedBundle("fluidaudio/Models/pocket-tts-old/flowlm_step.mlmodelc")
        try writeFinishedBundle("FluidAudio/Models/silero-vad/current.mlmodelc")
        try writeFinishedBundle("FluidAudio/Models/silero-vad/old.mlmodelc")
        try writeFinishedBundle("FluidAudio/Models/parakeet-old/encoder.mlmodelc")

        let removed = ModelLayout.removeUnusedModels(
            under: [
                root.appendingPathComponent("fluidaudio/Models"), root.appendingPathComponent("FluidAudio/Models"),
            ],
            keeping: [
                folder("fluidaudio/Models/pocket-tts", ["flowlm_step.mlmodelc"]),
                folder("FluidAudio/Models/silero-vad", ["current.mlmodelc"]),
            ]
        )

        XCTAssertEqual(
            Set(removed.map { relative($0).lowercased() }),
            [
                "fluidaudio/models/pocket-tts-old",
                "fluidaudio/models/silero-vad/old.mlmodelc",
                "fluidaudio/models/parakeet-old",
            ]
        )
        XCTAssertTrue(exists("fluidaudio/Models/pocket-tts/flowlm_step.mlmodelc/coremldata.bin"))
        XCTAssertTrue(exists("FluidAudio/Models/silero-vad/current.mlmodelc/coremldata.bin"))
    }

    func testCountsAnUnfinishedBundleAsMissing() throws {
        try writeFinishedBundle("silero-vad/finished.mlmodelc")
        try write("silero-vad/unfinished.mlmodelc/coremldata.bin")
        try write("silero-vad/vocab.json")
        let entries: Set<String> = ["finished.mlmodelc", "unfinished.mlmodelc", "vocab.json", "renamed_v2.mlmodelc"]

        XCTAssertEqual(
            ModelLayout.missingEntries(in: folder("silero-vad", entries)),
            ["renamed_v2.mlmodelc", "unfinished.mlmodelc"]
        )
    }

    func testBundleListedByItsFilesNeedsNoCompletionMarker() throws {
        try write("turn-detector/detector.mlmodelc/coremldata.bin")
        try write("turn-detector/detector.mlmodelc/model.mil")

        XCTAssertEqual(
            ModelLayout.missingEntries(in: turnDetectorFolder()), ["detector.mlmodelc/weights/weight.bin"])
    }

    func testSizesOnlyTheMissingFilesFromTheRemoteListing() async throws {
        try writeFinishedBundle("silero-vad/kept.mlmodelc")
        try write("silero-vad/renamed_v2.mlmodelc/coremldata.bin")
        let repo = Repo.vad.remotePath
        let pages: [String: [[String: Any]]] = [
            "tree/main": [
                ["path": "kept.mlmodelc", "type": "directory"],
                ["path": "renamed_v2.mlmodelc", "type": "directory"],
                ["path": "other.mlmodelc", "type": "directory"],
                ["path": "README.md", "type": "file", "size": 7],
            ],
            "tree/main/renamed_v2.mlmodelc": [
                ["path": "renamed_v2.mlmodelc/coremldata.bin", "type": "file", "size": 10],
                ["path": "renamed_v2.mlmodelc/weights", "type": "directory"],
            ],
            "tree/main/renamed_v2.mlmodelc/weights": [
                ["path": "renamed_v2.mlmodelc/weights/weight.bin", "type": "file", "size": 30_000_000]
            ],
        ]
        let server = ListingServer(repo: repo, pages: pages)

        let bytes = try await ModelLayout.missingBytes(
            in: [folder("silero-vad", ["kept.mlmodelc", "renamed_v2.mlmodelc"])], fetch: server.fetch)

        XCTAssertEqual(bytes, 30_000_000)
        XCTAssertEqual(
            server.requested,
            ["tree/main", "tree/main/renamed_v2.mlmodelc", "tree/main/renamed_v2.mlmodelc/weights"])
    }

    func testSizesPackByWhatItsDownloadFetches() async throws {
        let pack = try pocketTtsPack()
        let directory = pack.remoteDirectory
        let tree = "tree/main/\(directory)"
        let pages: [String: [[String: Any]]] = [
            tree: [
                ["path": "\(directory)/constants", "type": "directory"],
                ["path": "\(directory)/constants_bin", "type": "directory"],
                ["path": "\(directory)/cond_prefill.mlmodelc", "type": "directory"],
                ["path": "\(directory)/flowlm_step.mlmodelc", "type": "directory"],
                ["path": "\(directory)/flowlm_step_ane.mlmodelc", "type": "directory"],
                ["path": "\(directory)/flowlm_step.mlpackage", "type": "directory"],
                ["path": "\(directory)/manifest.json", "type": "file", "size": 5],
                ["path": "\(directory)/verify.wav", "type": "file", "size": 11],
            ],
            "\(tree)/constants_bin": [
                ["path": "\(directory)/constants_bin/tokenizer.model", "type": "file", "size": 400],
                ["path": "\(directory)/constants_bin/text_embed_table.bin", "type": "file", "size": 300],
                ["path": "\(directory)/constants_bin/bos_before_voice.bin", "type": "file", "size": 100],
            ],
            "\(tree)/cond_prefill.mlmodelc": [
                ["path": "\(directory)/cond_prefill.mlmodelc/coremldata.bin", "type": "file", "size": 1_000]
            ],
            "\(tree)/flowlm_step.mlmodelc": [
                ["path": "\(directory)/flowlm_step.mlmodelc/coremldata.bin", "type": "file", "size": 2_000]
            ],
        ]
        let server = ListingServer(repo: Repo.pocketTts.remotePath, pages: pages)

        let bytes = try await ModelLayout.missingBytes(in: [pack], fetch: server.fetch)

        XCTAssertEqual(bytes, 5 + 400 + 300 + 100 + 1_000 + 2_000)
        XCTAssertEqual(
            server.requested,
            [tree, "\(tree)/constants_bin", "\(tree)/cond_prefill.mlmodelc", "\(tree)/flowlm_step.mlmodelc"])

        try write("pack/manifest.json")
        try write("pack/constants_bin/tokenizer.model")
        let remaining = try await ModelLayout.missingBytes(in: [pack], fetch: server.fetch)
        XCTAssertEqual(remaining, 300 + 100 + 1_000 + 2_000)
    }

    func testSkipsTheListingWhenNothingIsMissing() async throws {
        try writeFinishedBundle("silero-vad/kept.mlmodelc")
        let fetch: HFTreeLister.Fetch = { _ in throw DownloadError.invalidResponse }

        let bytes = try await ModelLayout.missingBytes(
            in: [folder("silero-vad", ["kept.mlmodelc"])], fetch: fetch)

        XCTAssertEqual(bytes, 0)
    }

    func testSizesFolderFromRepositoryOutsideTheRepoEnum() async throws {
        try write("turn-detector/detector.mlmodelc/coremldata.bin")
        let tree = "tree/\(pinnedRevision)"
        let pages: [String: [[String: Any]]] = [
            tree: [
                ["path": "detector.mlmodelc", "type": "directory"],
                ["path": "README.md", "type": "file", "size": 7],
            ],
            "\(tree)/detector.mlmodelc": [
                ["path": "detector.mlmodelc/analytics", "type": "directory"],
                ["path": "detector.mlmodelc/weights", "type": "directory"],
                ["path": "detector.mlmodelc/coremldata.bin", "type": "file", "size": 10],
                ["path": "detector.mlmodelc/metadata.json", "type": "file", "size": 5],
                ["path": "detector.mlmodelc/model.mil", "type": "file", "size": 200],
            ],
            "\(tree)/detector.mlmodelc/weights": [
                ["path": "detector.mlmodelc/weights/weight.bin", "type": "file", "size": 16_000_000]
            ],
        ]
        let server = ListingServer(repo: otherRepository, pages: pages)

        let bytes = try await ModelLayout.missingBytes(in: [turnDetectorFolder()], fetch: server.fetch)

        XCTAssertEqual(bytes, 16_000_200)
        XCTAssertEqual(server.requested, [tree, "\(tree)/detector.mlmodelc", "\(tree)/detector.mlmodelc/weights"])
    }

    func testSizesEntriesUnderTheRemoteDirectory() async throws {
        let tree = "tree/\(pinnedRevision)/models/english"
        let pages: [String: [[String: Any]]] = [
            tree: [
                ["path": "models/english/voice.mlmodelc", "type": "directory"],
                ["path": "models/english/other.mlmodelc", "type": "directory"],
                ["path": "models/english/notes.txt", "type": "file", "size": 9],
            ],
            "\(tree)/voice.mlmodelc": [
                ["path": "models/english/voice.mlmodelc/coremldata.bin", "type": "file", "size": 30]
            ],
        ]
        let server = ListingServer(repo: otherRepository, pages: pages)
        let english = ModelFolder(
            url: root.appendingPathComponent("english"), requiredEntries: ["voice.mlmodelc"],
            remoteRepoPath: otherRepository, revision: pinnedRevision, remoteDirectory: "models/english")

        let bytes = try await ModelLayout.missingBytes(in: [english], fetch: server.fetch)

        XCTAssertEqual(bytes, 30)
        XCTAssertEqual(server.requested, [tree, "\(tree)/voice.mlmodelc"])
    }
}
