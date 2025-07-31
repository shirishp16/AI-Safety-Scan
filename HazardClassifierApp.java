import java.util.Scanner;

public class HazardClassifierApp {
    public static void main(String[] args) {
        // Directory where you saved your model via Python: model.save_pretrained("bert.model")
        String modelDirectory = "bert.model";

        Classifier classifier = new BertClassifier(modelDirectory);
        Scanner scanner = new Scanner(System.in);

        System.out.println("=== Hazard Classifier ===");
        System.out.println("Enter an observation (or 'exit' to quit):");

        while (true) {
            System.out.print("> ");
            String input = scanner.nextLine();
            if ("exit".equalsIgnoreCase(input.trim())) {
                break;
            }
            String label = classifier.classify(input);
            System.out.printf("Classification: %s%n%n", label);
        }

        System.out.println("Goodbye!");
        scanner.close();
    }
}
