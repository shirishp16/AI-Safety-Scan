/**
 * Simple interface for text classifiers.
 */
public interface Classifier {
    /**
     * Classify the given input string.
     * @param text the raw input observation
     * @return a label, e.g. "high-risk" or "low-risk"
     */
    String classify(String text);
}
