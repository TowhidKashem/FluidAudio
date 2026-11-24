import CoreML
import XCTest

@testable import FluidAudio

final class PocketTtsWordTimingTests: XCTestCase {

    func testTextAttentionScoresSoftmaxOverTheTextAndAverageTheHeads() throws {
        let attention = try MLMultiArray(shape: [1, 2, 8], dataType: .float32)
        for index in 0..<16 { attention[index] = 10 }
        let headLogits: [[Float]] = [[0, 0, 0], [0, log(2), 0]]
        for (head, logits) in headLogits.enumerated() {
            for (offset, logit) in logits.enumerated() {
                attention[head * 8 + 3 + offset] = NSNumber(value: logit)
            }
        }

        let scores = PocketTtsSynthesizer.textAttentionScores(attention, textRange: 3..<6)

        XCTAssertEqual(scores.count, 3)
        XCTAssertEqual(scores[0], 7 / 24, accuracy: 1e-5)
        XCTAssertEqual(scores[1], 10 / 24, accuracy: 1e-5)
        XCTAssertEqual(scores[2], 7 / 24, accuracy: 1e-5)
    }

    func testTextAttentionScoresAreEmptyWhenTheTextFallsOutsideTheCache() throws {
        let attention = try MLMultiArray(shape: [1, 2, 8], dataType: .float32)

        XCTAssertEqual(PocketTtsSynthesizer.textAttentionScores(attention, textRange: 6..<10), [])
    }

    func testPieceLooksUpTheTokenText() throws {
        let tokenizer = try SentencePieceTokenizer(
            modelData: SentencePieceTestModel.data(pieces: [
                ("<unk>", 0, .unknown),
                ("\u{2581}Hi", -1, .normal),
                ("!", -2, .normal),
            ]))

        XCTAssertEqual(tokenizer.piece(for: 1), "\u{2581}Hi")
        XCTAssertEqual(tokenizer.piece(for: 2), "!")
        XCTAssertNil(tokenizer.piece(for: 3))
    }
}
