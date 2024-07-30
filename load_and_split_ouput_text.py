import pandas as pd

# Function to load and split output text
def load_and_split_output_text(output_text_path):
    with open(output_text_path, 'r') as f:
        output_text = f.read()
    return [part.strip() for part in output_text.split('●') if part.strip()]

if __name__ == "__main__":
    # Output text file path
    output_text_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/3274_tree3_fine_tune.txt"
    output_csv_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/split_output.csv"

    # Load and split the output text
    output_text_parts = load_and_split_output_text(output_text_path)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({'output_text': output_text_parts})
    df.to_csv(output_csv_path, index=False)
    print(f"Output text has been split and saved to {output_csv_path}")
