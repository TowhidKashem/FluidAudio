import Foundation

public struct ModelFolder: Sendable {
    public let url: URL
    public let requiredEntries: Set<String>
    public let remoteRepoPath: String
    public let revision: String
    public let remoteDirectory: String
    let shouldSkip: (@Sendable (String) -> Bool)?

    public init(
        url: URL, requiredEntries: Set<String>, remoteRepoPath: String, revision: String, remoteDirectory: String
    ) {
        self.url = url
        self.requiredEntries = requiredEntries
        self.remoteRepoPath = remoteRepoPath
        self.revision = revision
        self.remoteDirectory = remoteDirectory
        self.shouldSkip = nil
    }

    init(
        url: URL, requiredEntries: Set<String>, repo: Repo, remoteDirectory: String? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) {
        self.url = url
        self.requiredEntries = requiredEntries
        self.remoteRepoPath = repo.remotePath
        self.revision = ModelRegistry.mapRevision(repo.remotePath, default: repo.revision)
        self.remoteDirectory = remoteDirectory ?? repo.subPath ?? ""
        self.shouldSkip = shouldSkip
    }
}

public enum ModelLayout {
    private static let logger = AppLogger(category: "ModelLayout")

    public static var speechModelsDirectory: URL {
        MLModelConfigurationUtils.defaultModelsDirectory()
    }

    public static func sileroVad() -> [ModelFolder] {
        [
            ModelFolder(
                url: MLModelConfigurationUtils.defaultModelsDirectory(for: .vad),
                requiredEntries: ModelNames.VAD.requiredModels,
                repo: .vad
            )
        ]
    }

    public static func parakeetEou160() -> [ModelFolder] {
        let root = speechModelsDirectory.appendingPathComponent("parakeet-eou-streaming", isDirectory: true)
        return [
            ModelFolder(
                url: root.appendingPathComponent(Repo.parakeetEou160.folderName, isDirectory: true),
                requiredEntries: ModelNames.ParakeetEOU.requiredModels,
                repo: .parakeetEou160
            )
        ]
    }

    public static func missingEntries(in folder: ModelFolder) -> [String] {
        ModelCache.incompleteFiles(at: folder.url, requiredFiles: folder.requiredEntries)
    }

    public static func missingBytes(in folders: [ModelFolder]) async throws -> Int {
        try await missingBytes(in: folders, fetch: HFTreeLister.fetch(using: ModelHub.session))
    }

    static func missingBytes(in folders: [ModelFolder], fetch: HFTreeLister.Fetch) async throws -> Int {
        var bytes = 0
        for folder in folders {
            let prefixes = missingEntries(in: folder).map {
                folder.remoteDirectory.isEmpty ? $0 : "\(folder.remoteDirectory)/\($0)"
            }
            guard !prefixes.isEmpty else { continue }
            let files = try await HFTreeLister.listTree(
                repoRemotePath: folder.remoteRepoPath,
                revision: folder.revision,
                startingAt: folder.remoteDirectory,
                include: { path, isDirectory in
                    if let shouldSkip = folder.shouldSkip { return !shouldSkip(path) }
                    return prefixes.contains { prefix in
                        path == prefix || path.hasPrefix(prefix + "/")
                            || (isDirectory && prefix.hasPrefix(path + "/"))
                    }
                },
                fetch: fetch
            )
            for file in files {
                let local = folder.url.appendingPathComponent(
                    ModelCache.localPath(for: file.path, subPath: folder.remoteDirectory))
                guard !FileManager.default.fileExists(atPath: local.path) else { continue }
                bytes += max(file.size, 0)
            }
        }
        return bytes
    }

    @discardableResult
    public static func removeUnusedModels(under roots: [URL], keeping folders: [ModelFolder]) -> [URL] {
        let rootKeys = Set(roots.map(key))
        var requiredNames: [String: Set<String>] = [:]
        var ancestors: Set<String> = []
        for folder in folders {
            if requiredNames[key(folder.url)] == nil {
                requiredNames[key(folder.url)] = []
            }
            for entry in folder.requiredEntries {
                var directory = folder.url
                for component in entry.split(separator: "/").map(String.init) {
                    requiredNames[key(directory), default: []].insert(component)
                    directory = directory.appendingPathComponent(component)
                }
            }
            var parent = folder.url.deletingLastPathComponent()
            while parent.pathComponents.count > 1, !rootKeys.contains(key(parent)) {
                ancestors.insert(key(parent))
                parent = parent.deletingLastPathComponent()
            }
        }

        var removed: [URL] = []
        var visited: Set<String> = []
        func visit(_ directory: URL) {
            let directoryKey = key(directory)
            guard visited.insert(directoryKey).inserted else { return }
            let names = requiredNames[directoryKey] ?? []
            for child in subdirectories(of: directory) {
                let childKey = key(child)
                if requiredNames[childKey] != nil || ancestors.contains(childKey) {
                    visit(child)
                } else if !names.contains(child.lastPathComponent), containsModelBundle(child) {
                    do {
                        try FileManager.default.removeItem(at: child)
                        removed.append(child)
                    } catch {
                        guard !ModelCache.isMissingFile(error) else { continue }
                        logger.warning(
                            "Failed to remove unused model folder \(child.path): \(error.localizedDescription)")
                    }
                }
            }
        }
        for root in roots where FileManager.default.fileExists(atPath: root.path) {
            visit(root)
        }
        return removed
    }

    private static func key(_ url: URL) -> String {
        let url = url.resolvingSymlinksInPath()
        return (try? url.resourceValues(forKeys: [.canonicalPathKey]))?.canonicalPath ?? url.path
    }

    private static func subdirectories(of directory: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.map { directory.appendingPathComponent($0) }.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
    }

    private static func containsModelBundle(_ directory: URL) -> Bool {
        if directory.pathExtension == "mlmodelc" { return true }
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        else { return false }
        for case let item as URL in enumerator where item.pathExtension == "mlmodelc" {
            return true
        }
        return false
    }
}

#if TTS
extension ModelLayout {
    public static func ttsModelsDirectory() throws -> URL {
        try TtsCacheDirectory.ensure().appendingPathComponent(KokoroAneResourceDownloader.modelsSubdirectory)
    }

    public static func kokoro() throws -> [ModelFolder] {
        let modelsDirectory = try ttsModelsDirectory()
        return [
            ModelFolder(
                url: try KokoroAneResourceDownloader.repositoryDirectory(
                    variant: .english, directory: modelsDirectory),
                requiredEntries: ModelNames.KokoroAne.requiredModels,
                repo: .kokoroAne
            ),
            ModelFolder(
                url: modelsDirectory.appendingPathComponent(Repo.kokoro.folderName),
                requiredEntries: ModelNames.G2P.requiredModels,
                repo: .kokoro
            ),
        ]
    }

    public static func pocketTts() throws -> [ModelFolder] {
        let repoDirectory = try pocketTtsRepoDirectory()
        let packDirectory = PocketTtsLanguage.english.repoSubdirectory
        let required = ModelNames.PocketTTS.requiredModels(precision: .fp16, placement: .gpu)
        return [
            ModelFolder(
                url: repoDirectory.appendingPathComponent(packDirectory),
                requiredEntries: required,
                repo: .pocketTts,
                remoteDirectory: packDirectory,
                shouldSkip: { PocketTtsResourceDownloader.shouldSkipAsset(at: $0, required: required) }
            ),
            ModelFolder(
                url: repoDirectory,
                requiredEntries: [ModelNames.PocketTTS.mimiEncoderFile],
                repo: .pocketTts,
                remoteDirectory: ""
            ),
        ]
    }

    public static func pocketTtsSmall(_ model: PocketTtsSmallModel) throws -> [ModelFolder] {
        [
            ModelFolder(
                url: try pocketTtsRepoDirectory().appendingPathComponent(model.repoSubdirectory),
                requiredEntries: Set(PocketTtsSmallDownloader.requiredFiles(for: model)),
                repo: .pocketTts,
                remoteDirectory: model.repoSubdirectory,
                shouldSkip: { PocketTtsSmallDownloader.shouldSkipAsset(at: $0) }
            )
        ]
    }

    public static func luxTts() throws -> [ModelFolder] {
        [
            ModelFolder(
                url: try ttsModelsDirectory().appendingPathComponent(Repo.luxtts.folderName),
                requiredEntries: ModelNames.LuxTts.requiredFiles(variant: ModelNames.LuxTts.defaultVariant),
                repo: .luxtts
            )
        ]
    }

    private static func pocketTtsRepoDirectory() throws -> URL {
        try TtsCacheDirectory.ensure()
            .appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
    }
}
#endif
